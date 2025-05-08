import pandas as pd
import numpy as np
import pickle
import os
from openai import OpenAI
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import  util
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import requests
from tcomplex import TComplEx
from transformers import DistilBertTokenizer
import time
import re
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')




client = OpenAI(api_key="sk-xxxxxxxxxxxxxxxx", base_url="https://xxxxxx.com")

def gpt4(prompt):
    response= client.chat.completions.create(
        model="your-model-name",
        messages=[
            {"role": "system", "content": "You are an expert at analysing to get a methodology from a case and you give a detailed methodology to solve that type of problem based on the case provided."},
            {"role": "user", "content": prompt}
        ],
            temperature=0.0
    )
    return response.choices[0].message.content


def chatgpt(prompt):
    retries = 0  
    while retries < 3:
        try:
            response = client.chat.completions.create(
                model="your-model-name",
                messages=[
                    {"role": "system", 
                     "content": "You are a decision-maker guiding an agent through a temporal knowledge graph. The goal is to help the agent navigate and query until it uncovers the correct answer. The agent will suggest various methods and queries, and based on the question's semantics and previous steps, you'll choose the best option. Let's do step by step."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:  
            error_message = str(e)
            if 'content_filter' in error_message:
                print("Error: Content filter triggered, retrying...")
                retries += 1  
                if retries < 3:
                    time.sleep(2)  
                else:
                    print("Max retries reached. Exiting.")
                    break  
            else:
                print(f"An error occurred: {error_message}")
                break 

def get_history_text(inference,history_dict,qid):
    history_text = []

    # Add question to history
    history_text.append(f"Question: {history_dict[qid]['question']['paraphrases'][0]}")

    # Process history data
    process_data = history_dict[qid]['process']
    for i, (action, response) in enumerate(zip(process_data['history_candidate_actions'], process_data['history_response'])):
        history_text.append(f"Candidate actions {i}: {action}")
        history_text.append(f"LLM Action {i}: {response}")
        
        retrieval_data = process_data.get('history_retrieval', [])
        retrieval_text = retrieval_data[i] if i < len(retrieval_data) else 'Wrong action, unable to get data.'
        history_text.append(f"Retrieval {i}: {retrieval_text}")

    # Add result to history
    if history_dict[qid]['result']:
        history_text.append('Correct!')
    else:
        try:
            answers = [inference.id2text[x] for x in history_dict[qid]['question']['answers']]
        except:
            answers = history_dict[qid]['question']['answers']
        history_text.append(f"Wrong! The answer is {answers}")

    # Print final history text
    return '\n'.join(history_text)


class MultiTQ():
    def __init__(self, path='../data/MultiTQ'):
        question, kg, text2id, id2text, rel2text, ts2id = self.get_multitq(path)
        self.question = question
        self.kg = kg
        self.id2text = id2text
        self.rel2text = rel2text
        self.ts2id = ts2id
        self.model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    def get_multitq(self, path='../data/MultiTQ'):
        with open(path + '/questions/processed_dev.json',encoding='utf-8') as f:
            question = json.load(f)
        kg = pd.read_csv(path + '/kg/full.txt', sep='\t', header=None)
        kg.columns = ['head', 'rel', 'tail', 'time']
        with open(path + '/kg/entity2id.json') as f:
            text2id = json.load(f)
            
        #id2text = {v: k.replace(',', '-') for k, v in text2id.items()}
        id2text = {v: k for k, v in text2id.items()}
        
        with open(path + '/kg/relation2id.json') as f:
            text2rel = json.load(f)
        rel2text = {v: k for k, v in text2rel.items()}

        with open(path + '/kg/ts2id.json') as f:
            ts2id = json.load(f)
        #id2ts = {v: k for k, v in ts2id.items()}

        return question, kg, text2id, id2text, rel2text, ts2id
    
    def relation_filter(self,q,rels):
        candidate_rels = ['Express_intent_to_meet_or_negotiate',
                         'Make_pessimistic_comment',
                         'Use_conventional_military_force',
                         'Provide_humanitarian_aid',
                         'Return,_release_person(s)',
                         'Make_a_visit',
                         'Sign_formal_agreement',
                         'Make_an_appeal_or_request',
                         'Express_intent_to_engage_in_diplomatic_cooperation_(such_as_policy_support)',
                         'Express_intent_to_cooperate',
                         'Praise_or_endorse',
                         'Reject',
                         'Criticize_or_denounce',
                         'Threaten',
                         'Engage_in_negotiation',
                         'Make_optimistic_comment',
                         'fight_with_small_arms_and_light_weapons',
                         'Investigate',
                         'Use_unconventional_violence',
                         'Host_a_visit',
                         'Discuss_by_telephone',
                         'Accuse']
        rels = [x for x in rels if x in candidate_rels]
        query_embedding = self.model.encode(q)
        passage_embedding = self.model.encode(rels)
        scores = util.dot_score(query_embedding, passage_embedding)
        indices = np.argsort(np.array(scores[0]))[::-1]
        top_indices = indices[:5]
        filtered_rels  = []
        for i in top_indices:
            filtered_rels.append(rels[i])
        return filtered_rels
    
    def __getitem__(self, qid):
        q = self.question[qid]['question']
        try:
            head_text = self.question[qid]['entities'][0].replace(' ','_').replace(',', '-')
        except:
            head_text = 'None'
        try:
            tail_text = self.question[qid]['entities'][1].replace(' ','_').replace(',', '-')
        except:
            tail_text = 'None'
        try:
            time = str(self.question[qid]['time'][0])
        except:
            time = 'no time constraints'

        event = 'None'
        event_text = 'None'
        
        rel_list = []
        for h in [head_text, tail_text]:
            rel_list += list(self.kg[self.kg['head'] == h].rel.unique())
            rel_list += list(self.kg[self.kg['tail'] == h].rel.unique())
        rel_text = list(set(rel_list))
        rel_text = self.relation_filter(q,rel_text )
        return q, head_text, rel_text, tail_text, time, event_text, self.question[qid]['answers']
    
def calculate_accuracy(history_dict):
    total_count = defaultdict(int)  
    correct_count = defaultdict(int)  

    total_answer_type_count = defaultdict(int)  
    correct_answer_type_count = defaultdict(int)  

    # 遍历 history_dict 统计数据
    for _, v in history_dict.items():
        question_type = v['question']['qtype']
        answer_type = v['question'].get('answer_type', 'unknown') 
        
        total_count[question_type] += 1
        total_answer_type_count[answer_type] += 1

        if v['result']: 
            correct_count[question_type] += 1
            correct_answer_type_count[answer_type] += 1

    total_questions = sum(total_count.values())
    total_correct = sum(correct_count.values())

    print("Accuracy by Question Type:")
    for question_type, total in total_count.items():
        accuracy = round(correct_count[question_type] / total, 3)
        print(f"{question_type}\tAccuracy: {accuracy}  Total: {total}")

    print("\nAccuracy by Answer Type:")
    for answer_type, total in total_answer_type_count.items():
        accuracy = round(correct_answer_type_count[answer_type] / total, 3)
        print(f"{answer_type}\tAccuracy: {accuracy}  Total: {total}")

    overall_accuracy = total_correct / total_questions
    print("\nOverall Accuracy:", overall_accuracy)

    return overall_accuracy


def evaluate_query(question, inference, qa_model, heads, tails, times, targets, f, top_m=10, k=10):
    qa_model.cuda()
    qa_model.eval()  # Set model to evaluation mode
    qa_model.lm_model.eval()
    with torch.no_grad():
        tokenized_question = tokenizer(question, return_tensors="pt", padding=True, truncation=True)
        question_embedding = qa_model.getQuestionEmbedding(
            tokenized_question['input_ids'].cuda(), 
            tokenized_question['attention_mask'].cuda()
        )

        heads = torch.from_numpy(np.array(heads)).cuda()
        tails = torch.from_numpy(np.array(tails)).cuda()
        times = torch.from_numpy(np.array(times)).cuda()
        targets = torch.tensor(targets).cuda()  # Convert targets to tensor

        head_embedding = qa_model.entity_time_embedding(heads)
        tail_embedding = qa_model.entity_time_embedding(tails)
        time_embedding = [qa_model._get_multi_transformered_time_embedding(times).unsqueeze(0)]
        time_embedding = torch.cat(time_embedding, dim=0)

        # Score the query result
        relation_embedding = qa_model.linear(question_embedding)
        relation_embedding1 = qa_model.dropout(qa_model.bn1(qa_model.linear1(relation_embedding)))
        relation_embedding2 = qa_model.dropout(qa_model.bn2(qa_model.linear2(relation_embedding)))

        scores_time = qa_model.score_time(head_embedding, tail_embedding, relation_embedding1)

        if f == 'get_tail_entity':
            scores_entity = qa_model.score_given_tail(head_embedding, tail_embedding, relation_embedding2, time_embedding)
        elif f == 'get_head_entity':
            scores_entity = qa_model.score_given_head(head_embedding, tail_embedding, relation_embedding2, time_embedding)
        else:
            scores_entity = torch.maximum(
                qa_model.score_given_head(head_embedding, tail_embedding, relation_embedding2, time_embedding),
                qa_model.score_given_tail(head_embedding, tail_embedding, relation_embedding2, time_embedding)
            )
        score = torch.cat((scores_entity, scores_time), dim=1)
    top_m_scores, top_m_indices = torch.topk(score[0], top_m, largest=True)
    top_m_entities = []
    for i in range(top_m):
        idx = top_m_indices[i].item()
        if(idx >= len(inference.id2text)):
            entity_name = inference.id2text.get(idx - len(inference.id2text), f"[Unknown:{idx}]")
        else:
            entity_name = inference.id2text.get(idx, f"[Unknown:{idx}]")
        entity_score = top_m_scores[i].item()
        top_m_entities.append((entity_name, entity_score))

    target_scores = score[0, targets]
    max_score, max_idx = target_scores.max(dim=0)
    max_score = max_score.item()
    max_entity_id = targets[max_idx].item()

    rank = (score[0] >= max_score).sum().item()
    top_k_threshold = k
    in_top_k = rank <= top_k_threshold
    entity_name = inference.id2text.get(max_entity_id, f"[Unknown:{max_entity_id}]")

    #print("\nTop-m entities by score:")
    #for idx, (name, s) in enumerate(top_m_entities, start=1):
    #    print(f"{idx}. {name} (Score: {s:.4f})")
    
    print(f"\nBest match from targets: {entity_name} (Score: {max_score:.4f})")
    print(f"Rank: {rank}/{len(score[0])}, In top {k}: {in_top_k}")

    return top_m_entities, in_top_k

def loadTkbcModel(tkbc_model_file):
    print('Loading tkbc model from', tkbc_model_file)
    x = torch.load(tkbc_model_file, map_location=torch.device("cpu"))
    num_ent = x['embeddings.0.weight'].shape[0]
    num_rel = x['embeddings.1.weight'].shape[0]
    num_ts = x['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = x['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size
    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(x)
    tkbc_model.cuda()
    print('Loaded tkbc model')
    return tkbc_model

def loadTkbcModel_complex(tkbc_model_file):
    print('Loading complex tkbc model from', tkbc_model_file)
    tcomplex_params = torch.load(tkbc_model_file)
    # complex_params = torch.load(tkbc_model_file)
    num_ent = tcomplex_params['embeddings.0.weight'].shape[0]
    num_rel = tcomplex_params['embeddings.1.weight'].shape[0]
    num_ts = tcomplex_params['embeddings.2.weight'].shape[0]
    print('Number ent,rel,ts from loaded model:', num_ent, num_rel, num_ts)
    sizes = [num_ent, num_rel, num_ent, num_ts]
    rank = tcomplex_params['embeddings.0.weight'].shape[1] // 2  # complex has 2*rank embedding size

    # now put complex params in tcomplex model

    # tcomplex_params['embeddings.0.weight'] = complex_params['embeddings.0.weight']
    # tcomplex_params['embeddings.1.weight'] = complex_params['embeddings.1.weight']
    torch.nn.init.xavier_uniform_(tcomplex_params['embeddings.2.weight'])  # randomize time embeddings

    tkbc_model = TComplEx(sizes, rank, no_time_emb=False)
    tkbc_model.load_state_dict(tcomplex_params)
    tkbc_model.cuda()
    print('Loaded complex tkbc model')
    return tkbc_model

def extract_action(response):
    pattern = r'(\$.*?\$)'  
    matches = re.findall(pattern, response)

    if matches:
        extracted = matches[0].replace(", ", ",") 
        return extracted
    return ""


def timesToIds(times, ts2id):
    output = []
    if times in ts2id.keys():
        return [ts2id[times] + 10488]
    else:
        keys = [x for x in ts2id.keys() if x.startswith(times)]
        output = [ts2id[key] + 10488 for key in keys]
        return output
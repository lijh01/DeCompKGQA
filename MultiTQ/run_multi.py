
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from utils import chatgpt,MultiTQ,calculate_accuracy, evaluate_query, loadTkbcModel, extract_action, timesToIds
from prompt import DECISION_INSTRUCTION_PHASE_1_entities, DECISION_INSTRUCTION_PHASE_1_no_entities, DECISION_INSTRUCTION_PHASE_2, CROSS_CHECKING_INSTRUCTION
import tqdm
from multitq import get_actions,Inference, enrich_entities_with_time
import pickle
from qa_baselines import QA_MultiQA
import torch

multitq = MultiTQ()
inference = Inference(multitq.kg,multitq.id2text,multitq.rel2text)
tkbc_model = loadTkbcModel('../models/tcomplex.ckpt')
QA_MultiQA = QA_MultiQA(tkbc_model).cuda()
filename = '../models/multitq.ckpt'

QA_MultiQA.load_state_dict(torch.load(filename))

inference = Inference(multitq.kg,multitq.id2text,multitq.rel2text,history=[],n_clusters = 10)


def decision(qid):
    inference.reset()
    attemp = 0
    entities = []
    kg_res = []
    max_attemp = 8
    q,head_text,rel_text,tail_text,time,event_text,answers = multitq[qid]
    print('Question {}:'.format(qid),q)
    hit = False
    error_actions = []
    while True:
        attemp+=1
        if attemp>=max_attemp:
            print('Unable to get answer!')
            inference.history.append([['Result: Unable to get answer!']])
            break
        choose = get_actions(head_text,rel_text,tail_text,time,event_text,entities,inference.LLM_rel_choose)
        choose = [action for action in choose if action not in error_actions]
        if len(choose) == 0:
            print('Unable to get answer!')
            inference.history.append([['Result: Unable to get answer!']])
            break
        print('-'*100)
        print('Step:',inference.tick)
        choose_text = '\n'.join(choose)
        
        history = ''
        for i in inference.history:
            history+= '\n'+ ' '.join(i)


        entities_str = ',  '.join([f"['{' '.join(map(str, item))}']" for item in entities])
        
        if(entities == []):
            prompt = DECISION_INSTRUCTION_PHASE_1_no_entities.replace('{question}',q)
            prompt = prompt.replace('{history}',history if len(history)>0 else 'None')
        else:
            prompt = DECISION_INSTRUCTION_PHASE_1_entities.replace('{question}',q)
            prompt = prompt.replace('{entities}', entities_str)
            prompt = prompt.replace('{history}',history if len(history)>0 else 'None')
        print('-'*100)

        end = False
        response = chatgpt(prompt)
        print(response)
        sub_question = response
        inference.history.append(['sub_question:', sub_question])

        prompt = DECISION_INSTRUCTION_PHASE_2.replace('{sub_question}', sub_question)
        prompt = prompt.replace('{actions}', choose_text)
        prompt = prompt.replace('{history}', history if len(history) > 0 else 'None')

        response = chatgpt(prompt)
        print(response)

        try:
            kg_res, entities, f, p = inference.take_action(response)
        except Exception as e:
            print('Error occurred during inference take_action:', e)
            continue

        if len(kg_res) == 0:
            print('No entities found!')
            error_actions.append(extract_action(response))
            inference.LLM_rel_choose = None
            continue

        if f == "answer":
            end = True
        
        else:
            try:
                if f == "get_head_entity" or f == "get_tail_entity":
                    text2id = inference.text2id
                    if f == "get_head_entity":
                        tails = [text2id[p[0]]]
                        heads = tails
                        targets = [text2id[item[0]] for item in kg_res]
                        if(p[2] == 'no time constraints'):
                            times = [10488]
                        else:
                            times = timesToIds(p[2], multitq.ts2id)
                    elif f == "get_tail_entity":
                        heads = [text2id[p[0]]]
                        tails = heads
                        targets = [text2id[item[2]] for item in kg_res]
                        if(p[2] == 'no time constraints'):
                            times = [10488]
                        else:
                            times = timesToIds(p[2], multitq.ts2id)
                    else:
                        heads = [text2id[p[0]]]
                        tails = [text2id[p[2]]]
                        targets = []
                        for item in kg_res:
                            ids = timesToIds(item[3], multitq.ts2id)  
                            targets.extend(ids)  
                        times = [10488]

                    top_m_entities, in_top_k = evaluate_query(
                        sub_question,
                        inference,
                        QA_MultiQA,
                        heads,
                        tails,
                        times,
                        targets,
                        f,
                        top_m=20,
                        k=1
                    )

                    if len(kg_res) == 0 or not in_top_k:
                        print("Fallback to LLM Judge...")

                        tkgqa_candidates = top_m_entities  
                        entities2 = enrich_entities_with_time(inference, tkgqa_candidates, response)
                        if not entities2: 
                            print("[INFO] No enriched entities found. Retaining original entities.")
                        else:
                            # llm judge
                            entities_str = ""
                            for entity in entities2:
                                head, rel, tail, time = entity
                                entities_str += f"{head} --{rel}--> {tail} (time: {time})\n"
                            prompt = CROSS_CHECKING_INSTRUCTION.replace('{sub_question}', sub_question)
                            prompt = prompt.replace('{choose_text}', choose_text)
                            prompt = prompt.replace('{query_str}', extract_action(response))
                            prompt = prompt.replace('{entities_str}', entities_str)
                            response = chatgpt(prompt)
                            print(response)

                            try:
                                kg_res, entities, f, p = inference.take_action(response)
                            except Exception as e:
                                print('Error occurred during inference take_action:', e)
                                continue
                else:
                    if len(entities) == 0:
                        inference.history = inference.history[:-1]
                        inference.history.append(['Entities list is empty! Retry to get entities and then take action again!'])
                    continue
            except Exception as e:
                print('Error occurred during inference take_action:', e)
                continue


            
        if end:
            print('The predicted answer is:',kg_res[0])            
            ans = multitq.question[qid]['answers']
            hit = inference.check_correctness(ans,kg_res[0])

            if hit:
                print('Correct!')
                inference.history.append([['Result: Correct!']])
                break
            else:
                print('Wrong! The correct answer is:',ans)
                inference.history.append([['Result: Wrong!']])
                break
                

    his = inference.get_history()
    inference.reset()
    return hit,his


history_dict = {}
start_id = 10000
end_id = 10005

for qid in tqdm.tqdm(range(start_id,end_id)):
    try:
        hit, his = decision(qid)
    except Exception as e:
        hit, his = False, []
        print(f"Error at qid: {qid}, Exception: {e}")
    history_dict[qid] = {'question':multitq.question[qid],'process':his,'result':hit}


with open('./history_dict.pkl','wb') as f:
    pickle.dump(history_dict,f)

calculate_accuracy(history_dict)





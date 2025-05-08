
action_templates = [
    'get_tail_entity({head},{rel},{time})',
    'get_head_entity({tail},{rel},{time})',
    'get_time({head},{rel},{tail})',
    'get_before({entities},{your specified time})',
    'get_after({entities},{your specified time})',
    'get_first({entities})',
    'get_last({entities})',
    'get_between({entities},{your specified start time},{your specified end time})',
    
]

DECISION_INSTRUCTION_PHASE_1_entities = '''
You are an expert query decomposer. Strictly follow these rules:

---

### **Input Context**:
- **Question**: {question}
- **Known Entities**: {entities}
- **Query History**: {history}

---

### **Task**:
Generate **ONLY** the next single-step sub-question OR final answer using **EXACTLY ONE** of the query patterns below.

---

### **Critical Constraints**:
1. **STRICTLY** follow the query patterns. **DO NOT create new question formats.**
2. Output MUST be either:
   - **A)** A natural language sub-question.
   OR
   - **B)** A final answer in the format: `Answer: [ONLY_THE_REQUIRED_INFORMATION]`
3. **DO NOT generate multi-step questions**.
4. **DO NOT generate reasoning, explanations, or intermediate steps.**
5. **DO NOT use function syntax like `$get_time()$`.**
6. **STRICTLY enforce time constraints if provided.**
7. **NEVER** ask for "relation" information. Never start with 'What is'. **Query MUST target entities or time.**
8. **When the SUBJECT (head entity) is known**, and the object is unknown (e.g., "Who did Ireland appeal to...?"), you MUST use the form:
   - `[CURRENT_HEAD] [RELATION] who?`
   (✅ NOT "Who [RELATION] [CURRENT_TAIL]?")
9. **If `Known Entities` is empty, DO NOT ask "Which entities from entities ... ?"** Instead, first retrieve the missing entities.
10. **Preserve original subject-object direction**:
   - If the question is of the form **"Who [VERB] [ENTITY]?"**, you MUST preserve that structure.
   - DO NOT reverse it to "[ENTITY] [VERB] who?"
---

### **Query Patterns** (Translate to natural language questions):

#### **Time-based Queries (ALWAYS include time constraint if provided)**:
- When did [HEAD] [RELATION] [TAIL]?
- Which entities from entities existed before [SPECIFIED_TIME]?
- Which entities from entities existed after [SPECIFIED_TIME]?
- Which entities from entities existed between [START_TIME] and [END_TIME]?

#### **Entity Queries**:
- Who [RELATION] [CURRENT_TAIL]?  ← Only if HEAD is unknown
- Who [RELATION] [CURRENT_TAIL] on [EXACT_TIME]?
- [CURRENT_HEAD] [RELATION] who?   ← Use this if HEAD is known
- [CURRENT_HEAD] [RELATION] who on [EXACT_TIME]?

#### **Specific Time Queries**:
- Which entity from entities occurred first?
- Which entity from entities occurred last?

---

### **Final Answer Format**:
- `Answer: [ONLY_THE_REQUIRED_INFORMATION]`

---

### **Output Format (NO DEVIATION ALLOWED)**:
- **If a sub-question is needed**: Output a natural language sub-question.
- **If the final answer is available**: Output `Answer: [ONLY_THE_REQUIRED_INFORMATION]`.
- When answering time, Match the **granularity** of the time expression in the original question:
  - If the question asks about a **year**, return only `YYYY` 
  - If it asks about a **month**, return `YYYY-MM` 
  - If it asks for a **specific date** or didn't mention, return `YYYY-MM-DD` 

'''

DECISION_INSTRUCTION_PHASE_1_no_entities = '''
You are an expert query decomposer for temporal knowledge graphs.

---

### Input:
- Question: {question}
- Query History: {history}

---

### Your Task:
Generate ONE of the following:
- A single-step natural language sub-question, OR
- A final answer in the format: `Answer: [ONLY_THE_REQUIRED_INFORMATION]`

---

### MUST-FOLLOW Rules:

1. Use ONLY the query patterns below. DO NOT create new formats.
2. DO NOT generate multi-step or complex questions.
3. DO NOT repeat the last query in the history.
4. DO NOT include reasoning, explanations, or code-style function names.
5. NEVER ask about "relations" or use "What is ...".
6. **If the subject (HEAD) is clear in the original question**, and the object is unknown (e.g., "Who did Ireland make an appeal to...?"), you MUST use:
   - `[CURRENT_HEAD] [RELATION] who?`
   (✅ NOT "Who [RELATION] [CURRENT_TAIL]?")
7. **Preserve original subject-object direction**:
   - If the question is of the form **"Who [VERB] [ENTITY]?"**, you MUST preserve that structure.
   - DO NOT reverse it to "[ENTITY] [VERB] who?"
- DO NOT assume the entity listed in `Known Entities` is always the TAIL; use the original question to infer subject-object direction first.
   - For example:
     - "Who rejected Japan?" → ✅ `Who rejected Japan?`
     - ❌ DO NOT generate `Japan rejected who?`
---
8.Never answer sub question with "before" or "after" or "between", if time constraint here is "before" or "after" or "between", ignore time constraints

### Query Patterns (must be followed exactly):

#### 🔹 Entity-based Questions:
- Who [RELATION] [CURRENT_TAIL]?  ← Only when HEAD is unknown
- Who [RELATION] [CURRENT_TAIL] on [EXACT_TIME]?
- [CURRENT_HEAD] [RELATION] who?   ← Use when HEAD is known
- [CURRENT_HEAD] [RELATION] who on [EXACT_TIME]?

> ✅ Only use time expressions like "on/in/at [TIME]" if the question provides a specific point in time.
> ❌ If the time is **relative** (e.g., "after 2000", "before Feb 21, 2012"), DO NOT include any time in entity questions.
> 🛠 Instead, ask this first: When did [HEAD] [RELATION] [TAIL]?

#### 🔹 Time-based Questions:
- When did [HEAD] [RELATION] [TAIL]?

---

### Output Format (NO DEVIATION ALLOWED):
- **If a sub-question is needed**: Output a natural language sub-question.
- **If the final answer is available**: Output `Answer: [ONLY_THE_REQUIRED_INFORMATION]`.

'''



DECISION_INSTRUCTION_PHASE_2 = '''
You are tasked with selecting the correct query function to answer the sub-question. Follow these instructions exactly:

---

### Step-by-Step Instructions:

1. Identify the main action or intent in the sub-question.
2. Determine the subject (HEAD) and object (TAIL) of the action.
3. **Select EXACTLY ONE query function from the Available Actions list. DO NOT invent, modify, or guess functions.**
4. For composite time reasoning (e.g., "after first", "between last"), apply a two-step process:
   - Step 1: Use a time filter function ($get_before$, $get_after$, or $get_between$) to obtain the candidate list.
   - Step 2: Use a temporal selector ($get_first$ or $get_last$) to refine the result.
5. If the final answer has been obtained, use the `$answer(...)$` function.

---

### Critical Rules:

- ✅ You should replace only these placeholders:
  - `{your specified time}`
  - `{your specified start time}`
  - `{your specified end time}`

- ❌ DO NOT substitute or alter these template fields:
  - `{entities}`
  - `{answer}`
  - `{CURRENT_HEAD}`
  - `{CURRENT_TAIL}`
  - `{RELATION}`

- Do NOT change the structure of the selected action.
- NEVER guess or fabricate a function if it is not listed under Available Actions.
- NEVER repeat the last function used in {history}.

- 🕓 **Time granularity rules**:
  - When the sub-question asks for a year (e.g., "which year"), answer in `YYYY`.
  - When the sub-question asks for a month (e.g., "which month" or "in which month"), answer in `YYYY-MM`.
  - When the sub-question asks for a specific date (e.g., "when"), answer in `YYYY-MM-DD`.
  - When sub question asks for "Answer: ", directly answer in the same format.
---

### Pattern-to-Function Mapping:

- Sub-questions of the form `Who [RELATION] [CURRENT_TAIL]?`
  → must map to `$get_head_entity(CURRENT_TAIL, RELATION, {your specified time})$`

- Sub-questions of the form `[CURRENT_HEAD] [RELATION] who?`
  → must map to `$get_tail_entity(CURRENT_HEAD, RELATION, {your specified time})$`

> These mappings are mandatory and cannot be overridden or swapped.

---

### Available Query Functions (for selection only):

- **Time-based Queries**:
  - `$get_time(HEAD, RELATION, TAIL)`
  - `$get_before({entities}, {your specified time})`
  - `$get_after({entities}, {your specified time})`
  - `$get_between({entities}, {your specified start time}, {your specified end time})`

- **Entity-based Queries**:
  - `$get_tail_entity(CURRENT_HEAD, RELATION, {your specified time})`
  - `$get_head_entity(CURRENT_TAIL, RELATION, {your specified time})`

- **Specific Time Queries**:
  - `$get_first({entities})`
  - `$get_last({entities})`

- **Answer Format**:
  - `$answer(YOUR_ANSWER)$`

---

### Output Format (NO DEVIATION):

Action:
<Copy exactly one action from the Available Actions above. Replace ONLY the allowed time fields if needed and choose granularity appropriately.>

Explanation:
<Briefly explain your choice, focusing on head/tail roles and time constraints.>

---

### Current Context:

Original Question: {question}
Sub-question: {sub_question}
Previous Actions: {history}
(end of previous actions)

Available Actions:
{actions}

Choose your next action from the list. If final answer, use `$answer(...)$`.
'''

CROSS_CHECKING_INSTRUCTION = """
You are acting as a query selection judge. A previous attempt to answer the sub-question failed, possibly due to an incorrect query formulation.

Your task is to **re-select the correct query function** from the list of candidates. You must analyze the sub-question, evaluate the previous query, and compare with the current candidates to make a more informed decision.

---

### What You Have:

- **Sub-question**:  
{sub_question}

- **Previously Used Query (Failed)**:  
{query_str}

- **Current Candidate Queries from Traditional QA System**:  
{tkgqa_candidates}

- **Available Options to Choose From**:  
{choose_text}

---

### Allowed Query Templates:

You are only allowed to choose a function from the following templates:

- **Time-based Queries**:
  - `$get_time(HEAD, RELATION, TAIL)$`

- **Entity-based Queries**:
  - `$get_tail_entity(CURRENT_HEAD, RELATION, {{your specified time}})$`
  - `$get_head_entity(CURRENT_TAIL, RELATION, {{your specified time}})$`

- **Answer Format**:
  - `$answer(YOUR_ANSWER)$`

---

### Instructions:

1. Carefully analyze the sub-question to identify:
   - The head and tail entities (or which one is known).
   - The main relation being asked about.
   - Any time-related constraints.

2. Read the previously failed query and identify what might be wrong:
   - Is the head/tail role swapped?
   - Is the relation incorrect or too broad?
   - Is the time condition missing or incorrect?

3. Evaluate the provided candidate queries (from traditional TKGQA methods). See if they include more suitable relations, time constraints, or clearer direction.

4. Based on all this, **choose the most reasonable query** from the `Available Options`.

---

### Output Format (Strictly Follow):

Action:
<Copy one action string from the Available Options above>

Explanation:
<In 1-3 sentences, explain why this new action is more appropriate than the previous query, and how it better answers the sub-question.>

---

### Important Constraints:

- ❌ Do not repeat the failed query exactly.
- ❌ Do not create new query formats outside of the available options.
- ✅ You must choose one from the provided options, even if imperfect.
- ✅ You must show reasoning clearly and concisely.
"""
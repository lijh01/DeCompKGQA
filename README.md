# DeCompKGQA

This is the code for the paper Temporal Knowledge Graph Question Answering via Sub-Question Decomposition and Cross-Checking.

## Architecture
![Architecture of DeCompKGQA](https://s21.ax1x.com/2025/05/08/pEL2jKO.png)
DeCompKGQA is a two-stage framework that leverages sub-question decomposition and a dedicated cross-checking module for iterative temporal reasoning. In the first stage, the input question is decomposed into a temporally constrained sub-question. In the second stage, the sub-question guides the selection of executable query templates for candidate retrieval, after which a cross-checking module compares the retrieved candidates against the outputs of an embedding-based TKGQA model on the same sub-question to determine the intermediate answer.

## Running the code

To run the code, you need to train from MultiTQ to get a multitq.ckpt and place it in the models folder first.

```bash
# cd MultiTQ
# python run_multi.py
 ```

## Acknowledgements

https://github.com/facebookresearch/tkbc

https://github.com/apoorvumang/CronKGQA

https://github.com/czy1999/MultiTQ

https://github.com/czy1999/ARI-QA
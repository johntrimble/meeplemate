import jq
import json

def parse_logs(log_file):
    result = []
    with open(log_file, 'r') as f:
        logs = f.readlines()
        for log in logs:
            try:
                result.append(json.loads(log))
            except:
                pass
    return result

def find_all_rag(logs):
    # jq 'select(.run_name=="thread-of-thought-rag-chain")|.run_id
    return jq.compile('select(.run_name=="thread-of-thought-rag-chain")|.run_id').input_values(logs).all()

def split_logs_by_run_name(run_name, logs):
    result = {}
    in_progress = set()
    for log in logs:
        if log["run_name"] == run_name:
            if log["event"] == "run_start":
                result[log["run_id"]] = [log]
                in_progress.add(log["run_id"])
            elif log["event"] == "run_end":
                run_logs = result[log["run_id"]]
                run_logs.append(log)
                del result[log["run_id"]]
                in_progress.remove(log["run_id"])
                yield (log["run_id"], run_logs)
        match_in_progress = set(log["parent_ids"]) & in_progress
        if len(match_in_progress) > 0:
            run_id = match_in_progress.pop()
            result[run_id].append(log)
    
def split_logs_by_rag_run(logs):
    return split_logs_by_run_name("thread-of-thought-rag-chain", logs)

def split_logs_self_consistency(logs):
    return split_logs_by_run_name("self-consistency", logs)

def find_logprobs_run_id(rag_run_id, logs):
    # jq 'select(.parent_ids|index("f48a996d-6ade-44b7-bb05-3ccbcee9986e"))|select(.run_name=="chat_model_output_with_logprobs" and .event=="run_end")|.run_id'
    return jq.compile('select(.parent_ids|index($rag_run_id))|select(.run_name=="chat_model_output_with_logprobs" and .event=="run_end")|.run_id', args={"rag_run_id": rag_run_id}).input_values(logs).first()

def find_cot_response_run_id(rag_run_id, logs):
    return jq.compile('select(.parent_ids|index($rag_run_id))|select((.run_name|contains("cot_response")) and .event=="run_end")|.run_id', args={"rag_run_id": rag_run_id}).input_values(logs).all()

def find_llm_output_cot(rag_run_id, logs):
    run_ids = find_cot_response_run_id(rag_run_id, logs)
    return jq.compile('select((.parent_ids - $run_ids) != .parent_ids)|select(.run_type=="llm" and .event=="run_end")', args={"run_ids": run_ids}).input_values(logs).first()
    # jq 'select(.parent_ids|index("151260f2-b543-4f34-aea0-e77ecb86f74f"))|select(.run_type=="llm" and .event=="run_end")'
    # return jq.compile('select(.parent_ids|index($run_id))|select(.run_type=="llm" and .event=="run_end")', args={"run_id": run_id}).input_values(logs).first()

def find_answer_extract_run_id(rag_run_id, logs):
    # jq 'select((.parent_ids|index("f48a996d-6ade-44b7-bb05-3ccbcee9986e")))|select(.run_name=="extract-answer" and .event=="run_end")|.run_id'
    return jq.compile('select((.parent_ids|index($rag_run_id)))|select(.run_name=="extract-answer" and .event=="run_end")|.run_id', args={"rag_run_id": rag_run_id}).input_values(logs).first()

def find_llm_output_answer_extract(rag_run_id, logs):
    run_id = find_answer_extract_run_id(rag_run_id, logs)
    # jq 'select(.parent_ids|index("9d07b756-9236-48ef-aa7b-e8207f278c9c"))|select(.run_type=="llm" and .event=="run_end")'
    return jq.compile('select(.parent_ids|index($run_id))|select(.run_type=="llm" and .event=="run_end")', args={"run_id": run_id}).input_values(logs).first()

def find_question(rag_run_id, logs):
    # jq 'select(.parent_ids|index("f48a996d-6ade-44b7-bb05-3ccbcee9986e"))|select(.run_name=="ChatPromptTemplate")|.inputs.question'
    return jq.compile('select(.parent_ids|index($rag_run_id))|select(.run_name=="ChatPromptTemplate")|.inputs.question', args={"rag_run_id": rag_run_id}).input_values(logs).first()

if __name__ == "__main__":
    import os

    # Parent directory of this file's directory
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs = parse_logs(f"{parent_directory}/app.log")
    for k, (consistency_run_id, consistency_logs) in enumerate(split_logs_self_consistency(logs)):
        for i, (run_id, rag_logs) in enumerate(split_logs_by_rag_run(consistency_logs)):
            result = {
                "self_consistency_run_id": consistency_run_id,
                "question": find_question(run_id, rag_logs),
                "cot_generation": find_llm_output_cot(run_id, rag_logs)["outputs"]["generations"][0][0],
                "extract_generation": find_llm_output_answer_extract(run_id, rag_logs)["outputs"]["generations"][0][0]
            }
            print(json.dumps(result))
            
    # with open(outputfile, 'w') as f:
    #     for i, rag_run_id in enumerate(rag_run_ids):
    #         result = {
    #             "question": find_question(rag_run_id, logs),
    #             "cot_generation": find_llm_output_cot(rag_run_id, logs)["outputs"]["generations"][0][0],
    #             "extract_generation": find_llm_output_answer_extract(rag_run_id, logs)["outputs"]["generations"][0][0]
    #         }
    #         print(i)
    #         # f.write(json.dumps(result))
    #         # f.write("\n")
    #         # f.flush()

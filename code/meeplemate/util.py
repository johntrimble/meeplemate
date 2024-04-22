import json

def dump_jsonl(obj_list, fp):
    for obj in obj_list:
        print(json.dumps(obj), file=fp)


def spit_jsonl(obj_list, f):
    opened_file = False
    if hasattr(f, "write"):
        fp = f
    else:
        fp = open(f, "w")
        opened_file = True
    
    try:
        for obj in obj_list:
            print(json.dumps(obj), file=fp)
    finally:
        if opened_file:
            fp.close()


def slurp_jsonl(f):
    opened_file = False
    if hasattr(f, "read"):
        fp = f
    else:
        fp = open(f, "r")
        opened_file = True
    
    try:
        return [json.loads(line) for line in fp]
    finally:
        if opened_file:
            fp.close()


def slurp_json(f):
    opened_file = False
    if hasattr(f, "read"):
        fp = f
    else:
        fp = open(f, "r")
        opened_file = True

    try:
        return json.load(fp)
    finally:
        if opened_file:
            fp.close()


def spit_json(obj, f):
    opened_file = False
    if hasattr(f, "write"):
        fp = f
    else:
        fp = open(f, "w")
        opened_file = True
    
    try:
        return json.dump(obj, fp)
    finally:
        if opened_file:
            fp.close()

import math

def print_exp_details(args):
    info = information(args)
    for i in info:
        print(i)
    write_info(args, info)
    
def write_info_to_accfile(filename, args):
    info = information(args)
    f = open(filename, "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()    
    
def write_info(args, info):
    f = open("./"+args.save+'/'+"a_info.txt", "w")
    for i in info:
        f.write(i)
        f.write('\n')
    f.close()
    
def information(args):
    info = []
    info.append('======================================')
    info.append(f'    Dataset: {args.dataset}')
    info.append(f'    Model: {args.model}')
    info.append(f'    Aggregation Function: {args.defence}')
    if math.isclose(args.malicious, 0) == False:
        info.append(f'    Attack method: {args.attack}')
        info.append(f'    Fraction of malicious agents: {args.malicious*100}%')
        info.append(f'    Attack Begin: {args.attack_begin}')
    else:
        info.append(f'    -----No Attack-----')
        
    info.append(f'    Number of agents: {args.num_users}')
    info.append(f'    Fraction of agents each turn: {int(args.num_users*args.frac)}({args.frac*100}%)')
    info.append(f'    Local batch size: {args.local_bs}')
    info.append(f'    Local epoch: {args.local_ep}')
    info.append(f'    Client_LR: {args.lr}')
    info.append(f'    Client_Momentum: {args.momentum}')
    info.append(f'    Global Rounds: {args.epochs}')
    info.append('======================================')
    return info
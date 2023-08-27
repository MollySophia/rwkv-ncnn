from rwkv.rwkv_v4neo import *
import os, types, copy
import torch
import ncnn
import numpy as np
from rwkv.utils import TOKENIZER
from rwkv_ncnn_custom_layer import *

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

#####################################################
args = types.SimpleNamespace()
args.RUN_DEVICE = "cpu"
args.FLOAT_MODE = "fp32"
args.vocab_size = 50277

args.n_layer = 32
args.n_embd = 2560
args.ctx_len = 1024
#####################################################

user = "User"
bot = "Bot"
interface = ":"

# The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.
# The following is a conversation between a highly knowledgeable and intelligent AI called {bot}, and a human called {user}. In the following interactions, {user} and {bot} converse in natural language, and {bot} do its best to answer {user}'s questions. {bot} is respectful, polite and inclusive. {bot} knows a lot, and always tells the truth.

init_prompt = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''
HELP_MSG = '''Commands:
say something --> chat with bot. use \\n for new line.
+alt --> alternate chat reply
+reset --> reset chat

+gen YOUR PROMPT --> free generation with any prompt. use \\n for new line.
+qa YOUR QUESTION --> free generation - ask any question (just ask the question). use \\n for new line.
+more --> continue last free generation (only for +gen / +qa)
+retry --> retry last free generation (only for +gen / +qa)

Now talk with the bot and enjoy. Remember to +reset periodically to clean up the bot's memory. Use RWKV-4 14B for best results.
This is not instruct-tuned for conversation yet, so don't expect good quality. Better use +gen for free generation.
'''

# Load Model
print(f'loading model...')
net = ncnn.Net()
net.register_custom_layer(
    "rwkv.rwkv_v4neo.RWKV_Time_Mixing", RWKV_Time_Mixing_layer_creator, RWKV_Time_Mixing_layer_destroyer
)
net.register_custom_layer(
    "rwkv.rwkv_v4neo.RWKV_Channel_Mixing", RWKV_Channel_Mixing_layer_creator, RWKV_Channel_Mixing_layer_destroyer
)

net.load_param("./output/model.ncnn.param")
net.load_model("./output/model.ncnn.bin")

ew = np.fromfile("./output/emb_weight.bin", dtype=np.float32).reshape(args.vocab_size, args.n_embd)
emb_weight = []
for i in range(args.vocab_size):
    emb_weight.append(ncnn.Mat(ew[i]).clone())

model_tokens = []
# current_state = []
# current_state = ncnn.Mat(args.n_layer * 5, args.n_embd)
# current_state.fill(0)
# for i in range(args.n_layer):
#     for j in range(5):
#         if j % 5 == 4:
#             data = -1e30
#         else:
#             data = 0
#         for k in range(args.n_embd):
#             current_state.append(data)
        
# current_state = ncnn.Mat(np.array(current_state, dtype=np.float32).reshape(args.n_layer * 5, args.n_embd))

current_state = torch.zeros(args.n_layer * 5, args.n_embd, device=args.RUN_DEVICE)
for i in range(args.n_layer):
    current_state[5*i+4] -= 1e30

########################################################################################################

def run_rnn(tokens, newline_adj = 0):
    global model_tokens, current_state, emb_weight
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        with net.create_extractor() as ex:
            ex.input("in0", emb_weight[model_tokens[-1]])
            # ex.input("in1", current_state)
            ex.input("in1", ncnn.Mat(current_state.numpy()).clone())

            _, _out = ex.extract("out0")
            _, _current_state = ex.extract("out1")
            current_state = torch.from_numpy(np.array(_current_state))
            if i == len(tokens) - 1:
                out = torch.from_numpy(np.array(_out, dtype=np.float32))

    out[0] = -999999999  # disable <|endoftext|>
    out[187] += newline_adj
    # if newline_adj > 0:
    #     out[15] += newline_adj / 2 # '.'
    return out

all_state = {}
def save_all_stat(srv, name, last_out):
    n = f'{name}_{srv}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    # all_state[n]['rnn'] = copy.deepcopy(np.array(current_state, dtype=np.float32))
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)

def load_all_stat(srv, name):
    global model_tokens, current_state
    n = f'{name}_{srv}'
    # current_state = ncnn.Mat((copy.deepcopy(all_state[n]['rnn'])))
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']

########################################################################################################

# Run inference
print(f'\nRun prompt...')

out = run_rnn(tokenizer.tokenizer.encode(init_prompt))
gc.collect()
torch.cuda.empty_cache()

save_all_stat('', 'chat_init', out)

srv_list = ['dummy_server']
for s in srv_list:
    save_all_stat(s, 'chat', out)

print(f'### prompt ###\n[{tokenizer.tokenizer.decode(model_tokens)}]\n')

def reply_msg(msg):
    print(f'{bot}{interface} {msg}\n')

def on_message(message):
    global model_tokens, current_state

    srv = 'dummy_server'

    msg = message.replace('\\n','\n').strip()
    if len(msg) > 1000:
        reply_msg('your message is too long (max 1000 tokens)')
        return

    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")
    if x_temp <= 0.2:
        x_temp = 0.2
    if x_temp >= 5:
        x_temp = 5
    if x_top_p <= 0:
        x_top_p = 0
    
    if msg == '+reset':
        out = load_all_stat('', 'chat_init')
        save_all_stat(srv, 'chat', out)
        reply_msg("Chat reset.")
        return

    elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg.lower() == '+more' or msg.lower() == '+retry':

        if msg[:5].lower() == '+gen ':
            new = '\n' + msg[5:].strip()
            # print(f'### prompt ###\n[{new}]')
            current_state = None
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

        elif msg[:4].lower() == '+qa ':
            out = load_all_stat('', 'chat_init')

            real_msg = msg[4:].strip()
            new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
            # print(f'### qa ###\n[{new}]')
            
            out = run_rnn(tokenizer.tokenizer.encode(new))
            save_all_stat(srv, 'gen_0', out)

            # new = f"\nThe following is an excellent Q&A session consists of detailed and factual information.\n\nQ: What is 3+5?\nA: The answer is 8.\n\nQ: {msg[9:].strip()}\nA:"
            # print(f'### prompt ###\n[{new}]')
            # current_state = None
            # out = run_rnn(tokenizer.tokenizer.encode(new))
            # save_all_stat(srv, 'gen_0', out)

        elif msg.lower() == '+more':
            try:
                out = load_all_stat(srv, 'gen_1')
                save_all_stat(srv, 'gen_0', out)
            except:
                return

        elif msg.lower() == '+retry':
            try:
                out = load_all_stat(srv, 'gen_0')
            except:
                return

        begin = len(model_tokens)
        out_last = begin
        for i in range(150):
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            if msg[:4].lower() == '+qa ':
                out = run_rnn([token], newline_adj=-1)
            else:
                out = run_rnn([token])
            
            xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
        print('\n')
        # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'gen_1', out)

    else:
        if msg.lower() == '+alt':
            try:
                out = load_all_stat(srv, 'chat_pre')
            except:
                return
        else:
            out = load_all_stat(srv, 'chat')
            new = f"{user}{interface} {msg}\n\n{bot}{interface}"
            # print(f'### add ###\n[{new}]')
            out = run_rnn(tokenizer.tokenizer.encode(new), newline_adj=-999999999)
            save_all_stat(srv, 'chat_pre', out)

        begin = len(model_tokens)
        out_last = begin
        print(f'{bot}{interface}', end='', flush=True)
        for i in range(999):
            if i <= 0:
                newline_adj = -999999999
            elif i <= 30:
                newline_adj = (i - 30) / 10
            elif i <= 130:
                newline_adj = 0
            else:
                newline_adj = (i - 130) * 0.25 # MUST END THE GENERATION
            token = tokenizer.sample_logits(
                out,
                model_tokens,
                args.ctx_len,
                temperature=x_temp,
                top_p_usual=x_top_p,
                top_p_newline=x_top_p,
            )
            out = run_rnn([token], newline_adj=newline_adj)

            xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
            if '\ufffd' not in xxx:
                print(xxx, end='', flush=True)
                out_last = begin + i + 1
            
            send_msg = tokenizer.tokenizer.decode(model_tokens[begin:])
            if '\n\n' in send_msg:
                send_msg = send_msg.strip()
                break
            
            # send_msg = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
            # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
            #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
            #     break
            # if send_msg.endswith(f'{bot}{interface}'):
            #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
            #     break

        # print(f'{model_tokens}')
        # print(f'[{tokenizer.tokenizer.decode(model_tokens)}]')

        # print(f'### send ###\n[{send_msg}]')
        # reply_msg(send_msg)
        save_all_stat(srv, 'chat', out)

print(HELP_MSG)

while True:
    msg = input(f'{user}{interface} ')
    if len(msg.strip()) > 0:
        on_message(msg)
    else:
        print('Erorr: please say something')


--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

---- Goal:  make sure that, for a given sequence of actions, the output loss defines a distribution that sums to 1.

local utils = require 'utils'
local misc = require 'misc'
local cmd = torch.CmdLine()
cmd:text('hello')
cmd:option('-mode', 'hack') 
cmd:option('-seed', 2)
cmd:option('-T', 15)
cmd:option('-eps', 1e-5)
cmd:option('-scale', 1.2)
cmd:option('-rnn_size', 3)
cmd:option('-emb_size', 2)
cmd:option('-depth', 2)
cmd:option('-tape_len', 6)
cmd:option('-out_tape_len', 4)
cmd:option('-num_out_symbols', 4)
cmd:option('-action_slowdown', 0.1)
cmd:option('-baseline_slowdown', 0.2)
cmd:option('-check_grad', true)
cmd:option('-efficient_mem', true)
cmd:text()
local params = cmd:parse(arg)
print ('params:')
print (params)
torch.manualSeed(params.seed)

mode = params.mode
if mode == 'hack' then
   require 'hack'
   print ("hack is used")
end
eps = params.eps
scale = params.scale

---- require the mem before requiring rlntm: 
local efficient_mem 
if params.efficient_mem then
   print ('using efficient mem!')
   efficient_mem = require 'efficient_mem'
else
   print ('using regular mem')
   require 'mem'
end


---- GLOBAL VARIABLES
CHECK_GRAD = true   --- a global variable to let everyone konw that we are checking the grad.
DO_NOT_ACCUMULATE_PROBS = true
local rlntm = require 'rlntm'

--- step 1:  instantiate the network
local T = params.T
local rnn_size = params.rnn_size
local emb_size = params.emb_size
local depth = params.depth
local tape_len = params.tape_len
local out_tape_len = params.out_tape_len
local num_out_symbols = params.num_out_symbols
local action_slowdown = params.action_slowdown
local baseline_slowdown = params.baseline_slowdown

local in_moves = {-1, 0, 1}
local mem_moves = {1, 0, -1}
local out_moves = {0, 1}

if out_tape_len > T then
    print (string.format('error:  out_tape_len (%d) exceeds T (%d), which will necessarily result in failure.', out_tape_len, T))
    return
end

local r = rlntm.create(T, rnn_size, emb_size, depth, num_out_symbols, in_moves, mem_moves, out_moves, action_slowdown, baseline_slowdown)

--- step 2:  create sequences of all possible actions.
local batch_size = num_out_symbols ^ out_tape_len
------------------ USING a smaller batchsize works, but only when:
------------------ 1) We don't use baselines (so we need to remove them)
------------------ 2) we don't use the (r_t + r_{t+1} + ...) trick.
--------------------- Instead, we replace each such partial sum with the complete sum (r_1 + r_2 + ...).  
------------------ Then, if 1 and 2 are removed, then using an arbitrary subset of the actions actually works!  Nice! :)

print ("batch_size = number of all output sequences = ", batch_size)

---- MORE GLOBAL VARIABLES
global_action_pos = 1
global_action_seq = torch.zeros(T*3, batch_size)
--- populate global action seq.
---- order of execution:  output, input, mem.  This is the order in which the global_action_seq is filled. 
for t = 1, T*3, 3 do
   global_action_seq[{t+0, {}}] = torch.random(#out_moves)    --- this is the order of execution in the computational graph.
   global_action_seq[{t+1, {}}] = torch.random(#in_moves)   --- as the graph changes, the order of the nodes may need to change too.
   global_action_seq[{t+2, {}}] = torch.random(#mem_moves)
end
----- After lots of manual verification, I came to the conclusion that global_action_seq is legit.
assert (global_action_seq:min() >= 1)
----- FINAL ACTION PROBS
global_action_probs = torch.zeros(batch_size):fill(1) 


local x, dx = r:getParameters()
local init_H = {}
for d = 1, 2*depth do
    init_H[d] = torch.randn(1, rnn_size):expand(batch_size, rnn_size):clone()
end

local input_tape = torch.randn(1, tape_len, emb_size):expand(batch_size, tape_len, emb_size):clone()
local input_mem = torch.randn(1, tape_len, rnn_size):expand(batch_size, tape_len, rnn_size):clone()
local output_tape = torch.randn(batch_size, out_tape_len)
local output_mask = torch.zeros(1, out_tape_len):expand(batch_size, out_tape_len):clone():fill(1) --- only ones for us.

for b = 1, batch_size do
    local a = b - 1
    for t = 1, out_tape_len do
        output_tape[{b, t}] = a % num_out_symbols + 1
        a = math.floor(a / num_out_symbols) 
    end
end

local function F(x)
   global_action_pos = 1   
   global_action_probs:fill(1)
   local input_mem_2
   if efficient_mem then
       efficient_mem.prepare(input_mem)
       input_mem_2 = torch.zeros(1)
   else
       input_mem_2 = input_mem
   end
   local inputs = {r, input_tape, input_mem_2, output_tape, output_mask, init_H, x, dx, CHECK_GRAD}
   local outputs = {rlntm.grad_and_loss(unpack(inputs))}
   local total_loss, grads, rlntm_outputs = unpack(outputs)
   local losses = rlntm_outputs[1]

   local visualize = require 'visualize'

   if efficient_mem then
       efficient_mem.prepare(input_mem)
   end
   local batch_id = 1
   global_action_pos = 1
   visualize.printTrace(inputs, batch_id)

   return losses
end


x:copy(torch.randn(x:size())):mul(scale)
print ('x:nElement() = ', x:nElement())

local l = F(x)
assert (l:nElement() == batch_size)
print ('loss probs sums = ', l:exp():sum())



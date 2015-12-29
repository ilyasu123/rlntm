
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua'
USE_GPU = false

local utils = require 'utils'
local misc = require 'misc'
local cmd = torch.CmdLine()
cmd:text('hello')
cmd:option('-mode', 'hack') 
cmd:option('-seed', 1)
cmd:option('-T', 3)
cmd:option('-eps', 1e-3)
cmd:option('-scale', 1.2)
cmd:option('-rnn_size', 2)
cmd:option('-emb_size', 2)
cmd:option('-depth', 1)
cmd:option('-tape_len', 4)
cmd:option('-out_tape_len', 1)
cmd:option('-num_out_symbols', 4)
cmd:option('-action_slowdown', 0.1)
cmd:option('-baseline_slowdown', 0.2)
cmd:option('-check_grad', true)
cmd:option('-efficient_mem', true)
cmd:option('-per_step_gain', -0.4)
cmd:option('-entropy_gain', -0.1)
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
DO_NOT_ACCUMULATE_PROBS = false
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
local per_step_gain = params.per_step_gain
local entropy_gain = params.entropy_gain
local separate_baseline = 1

local in_moves = {-1, 0, 1}
local mem_moves = {1, 0, -1}
local out_moves = {0, 1}

if out_tape_len > T then
    print (string.format('error:  out_tape_len (%d) exceeds T (%d), which will necessarily result in failure.', out_tape_len, T))
    return
end

print ('per_step_gain = ', per_step_gain)
print ('entropy_gain = ', entropy_gain)
local r = rlntm.create(T, T, rnn_size, emb_size, depth, num_out_symbols, in_moves, mem_moves, out_moves, action_slowdown, baseline_slowdown, per_step_gain, entropy_gain, separate_baseline)

--- step 2:  create sequences of all possible actions.
local batch_size = (#in_moves * #mem_moves * #out_moves) ^ T  
------------------ USING a smaller batchsize works, but only when:
------------------ 1) We don't use baselines (so we need to remove them)
------------------ 2) we don't use the (r_t + r_{t+1} + ...) trick.
--------------------- Instead, we replace each such partial sum with the complete sum (r_1 + r_2 + ...).  
------------------ Then, if 1 and 2 are removed, then using an arbitrary subset of the actions actually works!  Nice! :)

print ("batch_size = number of all possible actions = ", batch_size)

---- MORE GLOBAL VARIABLES
global_action_pos = 1
global_action_seq = torch.zeros(T*3, batch_size)
--- populate global action seq.
---- order of execution:  output, input, mem.  This is the order in which the global_action_seq is filled. 
for b = 1, batch_size do
    local a = b-1
    for t = 1, T*3, 3 do
       local in_action = a % #in_moves + 1
       a = math.floor(a / #in_moves)
       local mem_action = a % #mem_moves + 1
       a = math.floor(a / #mem_moves)
       local out_action = a % #out_moves + 1
       a = math.floor(a / #out_moves)
       global_action_seq[{t+0, b}] = out_action    --- this is the order of execution in the computational graph.
       global_action_seq[{t+1, b}] = in_action   --- as the graph changes, the order of the nodes may need to change too.
       global_action_seq[{t+2, b}] = mem_action
   end
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
local output_tape = torch.randn(1, out_tape_len):random(num_out_symbols):expand(batch_size, out_tape_len):clone()
local output_mask = torch.zeros(1, out_tape_len):expand(batch_size, out_tape_len):clone():fill(1)

---rlntm.grad_and_loss(r, input_tape, input_mem, output_tape, output_mask, init_H, x, dx)

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
   local loss = rlntm.grad_and_loss(r, input_tape, input_mem_2, output_tape, output_mask, init_H, x, dx, CHECK_GRAD)
   assert (global_action_pos == 3*T + 1)
   return loss
end

local function G(x)
   global_action_pos = 1
   global_action_probs:fill(1)
   local input_mem_2
   if efficient_mem then
       efficient_mem.prepare(input_mem)
       input_mem_2 = torch.zeros(1)
   else
       input_mem_2 = input_mem
   end
   local loss, zero_one_loss, grads = rlntm.grad_and_loss(r, input_tape, input_mem_2, output_tape, output_mask, init_H, x, dx, CHECK_GRAD)
   assert (global_action_pos == 3*T + 1)
   return grads.dx:clone()
end

x:copy(torch.randn(x:size())):mul(scale)
print ('x:nElement() = ', x:nElement())

local l = F(x)

print ('loss = ', l)
print ('CHECK_GRAD:  sum of probs = ', global_action_probs:sum())
print ('probability error = ', 1 - global_action_probs:sum())
assert (global_action_pos  == 3*T + 1)


if  params.check_grad == true then
    local g = G(x)
    local g2 = torch.zeros(g:size())
    for i = 1, x:nElement() do
        local x_i = x[i]
        x[i] = x_i + eps
        local l1 = F(x)
        x[i] = x_i - eps
        local l2 = F(x)
        g2[i] = (l1 - l2) / (2*eps)
        x[i] = x_i
        print (string.format('%04d:  comp = %+12.8f      numeric = %+12.8f     rel = %+12.8f   name=%s', 
                             i, g[i], g2[i], (g[i]-g2[i])/g2[i], misc.find_module_by_param(r, i)))
    end
end

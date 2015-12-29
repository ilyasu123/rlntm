
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
local utils = require 'rlntm_pkgs.utils'
local misc = require 'rlntm_pkgs.misc'

local ExpandVec, parent = torch.class('nn.ExpandVec', 'nn.Module')
function ExpandVec:__init(dim)
   self.dim = dim
end
function ExpandVec:updateOutput(input)
   local tt = input
   assert (utils.istensor(tt))
   assert (tt:nDimension() == 2)
   assert (tt:size(2) == 1)
   self.output = tt:expand(tt:size(1), self.dim)
   return self.output
end 
function ExpandVec:updateGradInput(input, gradOutput) 
   self.gradInput = gradOutput:sum(2)
   return self.gradInput
end

local EltwiseVecTimesMul, parent = torch.class('nn.EltwiseVecTimesMul', 'nn.Module')
function EltwiseVecTimesMul:__init()
end

function EltwiseVecTimesMul:updateOutput(input)
   assert (#input == 2)
   local u, v = unpack(input)
   --- dim u = batch_size * 1 
   --- dim v = batch_size * N
   local batch_size_, one_ = unpack(u:size():totable())
   local batch_size, N = unpack(v:size():totable())
   assert (batch_size == batch_size_ and one_ == 1)
   self.output = torch.cmul(u:expand(batch_size, N), v)
   return self.output
end

function EltwiseVecTimesMul:updateGradInput(input, gradOutput)
   local u, v = unpack(input)
   local batch_size, N = unpack(v:size():totable())
   local batch_size_, one_ = unpack(u:size():totable())
   assert (batch_size == batch_size_ and one_ == 1)
   local grad_v  = torch.cmul(gradOutput, u:expand(batch_size, N))
   local grad_u_ = torch.cmul(gradOutput, v)
   local grad_u  = grad_u_:sum(2)
   self.gradInput = {grad_u, grad_v}
   return self.gradInput
end


local OutputActionConstraints, parent = torch.class('nn.OutputActionConstraints','nn.Module')
function OutputActionConstraints:__init(action_map, T, t)
    parent.__init(self)
    self.action_map = action_map
    assert (#action_map == 2)
    assert (action_map[1] == 0 and action_map[2] == 1)
    self.T = T
    self.t = t
end

function OutputActionConstraints:updateOutput(input)
    assert (#input == 2)
    local mask, ptr = unpack(input)
    local batch_size, input_len = unpack(mask:size():totable())
    self.output:resize(batch_size, 2):zero()
    local remaining_lengths = mask:sum(2):add(-1, ptr):add(1) -- of size batch_size
    local remaining_steps = self.T - self.t + 1 
    local huge = 1e10
    for b = 1, batch_size do
        if remaining_lengths[{b,1}] == 0 then
            self.output[{b,2}] = -huge ---- once you reached the end, you must stay put. Action 2 is disallowed.
        elseif remaining_steps == remaining_lengths[{b,1}] then
            self.output[{b,1}] = -huge ---- it is now forbidden to stay.  You must go.
        elseif remaining_steps < remaining_lengths[{b,1}] then
            -- then you must've violated something.  We wouldn't vo
            print ('self.T = ', self.T, 'self.t = ', self.t)
            print ('remaining_lengths[{b,1}] =  ', remaining_lengths[{b,1}])
            print ('mask = ', mask[1])
            assert (CHECK_GRAD == true) --- OR:  importance sampling == ture.
            assert (global_action_probs[b] == 0) --- we are in a zero probability event.
        end
    end
    return self.output
end

function OutputActionConstraints:updateGradInput(input, gradOutput)
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end

local RemainingOutputs, parent = torch.class('nn.RemainingOutputs','nn.Module')
function RemainingOutputs:__init()
    parent.__init(self)
end

function RemainingOutputs:updateOutput(input)
    assert (#input == 2)
    local out_mask, out_pos = unpack(input)
    local out_lengths = out_mask:sum(2)
    self.output:resize(out_lengths:size()):add(out_lengths, -1, out_pos):add(1) ---- TODO:  make sure that the "add(1)" part is necessary and correct 

    --- sadly self.output can be negative. I kinda don't like that. Thus:
    local negs = torch.le(self.output, 0):double()
    self.output:cmul(negs:mul(-1):add(1))
    assert (self.output:min() >= 0)

    return self.output 
end

function RemainingOutputs:updateGradInput(input, gradOutput)
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end



local ZeroOneLoss, parent = torch.class('nn.ZeroOneLoss','nn.Module')

function ZeroOneLoss:__init(t)
    parent.__init(self)
    self.name = "ZeroOneLoss"
end

function ZeroOneLoss:updateOutput(input)
   assert (#input == 3)
   local pred, targ, mask = unpack(input)
   assert (pred:nDimension() == 2)
   if targ:nDimension() == 1 then targ = targ:view(targ:nElement(), 1) end
   if mask:nDimension() == 1 then mask = mask:view(mask:nElement(), 1) end
   assert (targ:nDimension() == 2)
   assert (mask:nDimension() == 2)
   local pred_val, pred_arg = torch.max(pred, 2) 
   assert (misc.same_size(pred_arg:size(), targ:size()))
   assert (misc.same_size(targ:size(), mask:size()))
   self.output = torch.ne(pred_arg:double(), targ):double():cmul(mask)
   assert (self.output:max() <= 1)
   assert (self.output:min() >= 0)
   return self.output
end


function ZeroOneLoss:updateGradInput(input, gradOutput)
   if false then --mode == "hack" then
       self.gradInput = "nil" 
   else
       self.gradInput = misc.clone_zero(input)
   end
   assert (self.gradInput ~= nil)
   return self.gradInput
end





--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
local Rectifier, parent = torch.class('nn.Rectifier', 'nn.Module')
function Rectifier:updateOutput(input)
    self.output = torch.cmul(torch.gt(input, 0):double(), input)	 
    return self.output
end

function Rectifier:updateGradInput(input, gradOutput)
    self.gradInput = torch.cmul(gradOutput, torch.gt(input, 0):double())
    return self.gradInput 
end


local ConstAdd, parent = torch.class('nn.ConstAdd', 'nn.Module')
function ConstAdd:__init(s)
    self.s = s
end

function ConstAdd:updateOutput(input)
    self.output = torch.add(input, self.s)
    return self.output
end

function ConstAdd:updateGradInput(input, gradOutput)
    self.gradInput = gradOutput
    return self.gradInput
end


local ConstMul, parent = torch.class('nn.ConstMul', 'nn.Module')
function ConstMul:__init(s)
    self.s = s
end

function ConstMul:updateOutput(input)
    self.output = torch.mul(input, self.s)
    return self.output
end

function ConstMul:updateGradInput(input, gradOutput)
    self.gradInput = torch.mul(gradOutput, self.s)
    return self.gradInput
end

local ExactLogSoftMax, parent = torch.class('nn.ExactLogSoftMax', 'nn.Module')
function ExactLogSoftMax:updateOutput(input)
    assert (input:nDimension() == 2)
    local m = input:max(2):expand(input:size(1), input:size(2))
    local input_minus_max = input - m

    local u = torch.exp(input_minus_max)
    local z_minus_m = u:sum(2):expand(input:size(1), input:size(2))
    local log_z = torch.log(z_minus_m)
    self.output = input_minus_max - log_z
    return self.output
end

function ExactLogSoftMax:updateGradInput(input, gradOutput)
    local m = input:max(2):expand(unpack(input:size():totable()))
    local input_minus_max = input - m
    local u = torch.exp(input_minus_max)
    local z_minus_m = u:sum(2):expand(unpack(input:size():totable()))
    local probs = torch.cdiv(u, z_minus_m)

    local part_1 = gradOutput 
    local part_2 = torch.cmul(gradOutput:sum(2):expand(unpack(gradOutput:size():totable())), probs)
    self.gradOutput = part_1 - part_2
    
    return self.gradOutput    
end

local utils_2 = {}
function utils_2.cloneManyTimes(net, T)
  -- from https://github.com/wojciechz/learning_to_execute/blob/master/utils/utils.lua
  local clones = {}
  local params, gradParams = net:parameters()
  if params == nil then
    params = {}
  end
  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end
  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)
  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGradParams = clone:parameters()
    local cloneParamsNoGrad
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    if paramsNoGrad then
      cloneParamsNoGrad = clone:parametersNoGrad()
      for i =1,#paramsNoGrad do
        cloneParamsNoGrad[i]:set(paramsNoGrad[i])
      end
    end
    clones[t] = clone
    collectgarbage()
  end
  mem:close()
  return clones
end
return utils_2
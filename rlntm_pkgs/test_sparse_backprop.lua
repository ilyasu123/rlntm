
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua'

require 'Summable'
require 'nn'
require 'nngraph'
require 'strict'
local utils = require 'utils'


local AccBP, parent = torch.class('nn.AccBP', 'Summable')
function AccBP:__init(val)
    parent.__init(self)
    self.value = val
end

function AccBP:size()
    return self.value:size()
end

function AccBP:clone()
    return nn.AccBP(self.value:clone())
end

function AccBP:resizeAs(accbp)
    self.value:resizeAs(accbp.value)
end

function AccBP:fill(scalar)
    self.value:fill(scalar)
end

function AccBP:add(accbp)
    self.value:add(accbp.value)
end



local Accumulator, parent = torch.class('nn.Accumulator','nn.Module')
function Accumulator:__init()
    parent.__init(self)
end

function Accumulator:updateOutput(input)
    assert (#input == 2)
    local acc, extra = unpack(input)
    assert (Summable.is_summable(acc))
    acc.value:add(extra)
    self.output = acc
    return self.output
end

function Accumulator:updateGradInput(input, gradOutput)
    local acc, extra = unpack(input)
    assert (Summable.is_summable(gradOutput))

    local gradExtra = gradOutput.value
    self.gradInput = {gradOutput, gradExtra}
    return self.gradInput
end

local ExtractAcc, parent = torch.class('nn.ExtractAcc', 'nn.Module')
function ExtractAcc:__init()
    parent.__init(self)
end

function ExtractAcc:updateOutput(input)
    self.output = input.value
    return self.output
end

function ExtractAcc:updateGradInput(input, gradOutput)
    assert (utils.istensor(gradOutput))
    self.gradInput = nn.AccBP(gradOutput)
    return self.gradInput
end


local BoxAcc, parent = torch.class('nn.BoxAcc', 'nn.Module')
function BoxAcc:__init()
    parent.__init(self)
end

function BoxAcc:updateOutput(input)
    return nn.AccBP(input)
end         

function BoxAcc:updateGradInput(input, gradOutput)
   assert (Summable.is_summable(gradOutput))
   return gradOutput.value
end


---- OK. time for the test.
local utils = require 'utils'
local cmd = torch.CmdLine()
cmd:text('hello, testing backprop with funky objects.')
cmd:option('-seed', 1)
cmd:option('-N', 3)
cmd:option('-m', 2)
cmd:option('-n', 3)
cmd:text()
local params = cmd:parse(arg)
torch.manualSeed(params.seed)
print (params)

local N = params.N
local inputs = {}
for i = 1, N do
     inputs[i] = nn.Identity()()
end 
local acc = nn.BoxAcc()(inputs[1])
for i = 2, N do
   acc = nn.Accumulator(){acc, inputs[i]}
end
local output = nn.ExtractAcc()(acc)
local g = nn.gModule(inputs, {output})

local test_inputs = {}
local n = params.n
local m = params.m
local s = torch.zeros(n, m)
for i = 1, N do
    test_inputs[i] = torch.randn(n, m)
    s:add(test_inputs[i])
end


local test_outputs = g:forward(test_inputs)
print ('test_outputs = ', test_outputs)
print ('s            = ', s)
local gradInput = g:backward(test_inputs, test_outputs)
print ('gradInput = ', gradInput)
for i = 1, N do 
    print ('gradInput[', i, ']')
    print (gradInput[i])
end

--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

package.path = package.path .. ';rlntm_pkgs/?.lua'

require 'totem'
require 'mem'
require 'output'

local test = {}
local tester = totem.Tester()


local scale = 0.1

local batch_size = 2
local len = 2
local dim = 2

local pos = torch.zeros(batch_size):random(len)

function test.read()
   local node = nn.ReadMem()
   local input = {torch.randn(batch_size, len, dim):mul(scale),
                  torch.randn(batch_size):random(len):add(0.2)}
   local output = node:updateOutput(input)
   totem.nn.checkGradients(tester, node, input)
end


function test.write()
   local node = nn.WriteMem()
   local input = {torch.randn(batch_size, len, dim):mul(scale),
                  torch.randn(batch_size):random(len):add(0.2),
                  torch.randn(batch_size, dim):mul(scale),
                  torch.randn(batch_size, dim):mul(scale)}
   local output = node:updateOutput(input)
   totem.nn.checkGradients(tester, node, input)
end

function test.eltwiseMult()
   local node = nn.EltwiseVecTimesMul()
   local input = {torch.randn(batch_size,   1):mul(scale),
                  torch.randn(batch_size, dim):mul(scale)}
   local output = node:updateOutput(input)
   totem.nn.checkGradients(tester, node, input)
end


function test.ExpandVec()
   local node = nn.ExpandVec(dim)
   local input = torch.randn(batch_size,   1):mul(scale)
   local output = node:updateOutput(input)
   totem.nn.checkGradients(tester, node, input)
end


tester:add(test):run()

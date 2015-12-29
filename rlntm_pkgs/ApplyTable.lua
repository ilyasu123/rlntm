
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nn'
local misc = require 'rlntm_pkgs.misc'

local ApplyTable, parent = torch.class('nn.ApplyTable','nn.Module')
function ApplyTable:__init(table)
    parent.__init(self)
    self.table = table
end

function ApplyTable:updateOutput(input)
   self.output:resize(input:size())
   local ii = input:view(input:nElement())
   local oo = self.output:view(input:nElement())
   for i = 1, input:nElement() do
       if (self.table[ii[i]] == nil) then
           print (input)
           print (self.table)
       end
   
       oo[i] = self.table[ii[i]]
   end
   return self.output
end

function ApplyTable:updateGradInput(input, gradOutput)         
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end

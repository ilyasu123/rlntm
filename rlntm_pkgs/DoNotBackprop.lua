
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local misc = require 'rlntm_pkgs.misc'
local DoNotBackprop, parent = torch.class('nn.DoNotBackprop','nn.Module')
function DoNotBackprop:__init()
    parent.__init(self)
end

function DoNotBackprop:updateOutput(input)
    self.output = input
    return self.output 
end

function DoNotBackprop:updateGradInput(input, gradOutput)
   if mode == "hack" then
       self.gradInput = "nil"
   else
       self.gradInput = misc.clone_zero(input)
   end
   return self.gradInput
end         



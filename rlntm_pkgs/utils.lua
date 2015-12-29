
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local utils = {}

function utils.istensor(x)
   if torch.typename(x) and torch.typename(x):find('Tensor') then
      return true
   else
      return false
   end
end

function utils.iscudatensor(x)
   if torch.typename(x) and torch.typename(x):find('CudaTensor') then
      return true
   else
      return false
   end
end


function utils.istorchclass(x)
   return type(x) == 'table' and torch.typename(x)
end

function utils.istable(x)
   return type(x) == 'table' and not torch.typename(x)
end

return utils



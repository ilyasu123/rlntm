
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local Summable = torch.class('Summable')

function Summable.is_summable(x)
     if type(x) == "userdata" or type(x) == "table" then 
         return x.type == '__summable__4.pu;!@#'
     else
         return false
     end
end

function Summable:__init()
     self.type = '__summable__4.pu;!@#'
end

function Summable:size()

end

function Summable:clone()
   
end

function Summable:resizeAs(what)

end

function Summable:fill(scalar)

end

function Summable:add(what)
    
end

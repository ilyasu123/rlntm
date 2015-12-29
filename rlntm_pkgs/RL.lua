
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local RL = {}
require 'rlntm_pkgs.RLSample'
require 'rlntm_pkgs.RLForce'
if CHECK_GRAD == true 
then RL.Sample = nn.RLForce
else RL.Sample = nn.RLSample
end

return RL 

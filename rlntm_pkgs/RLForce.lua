
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

local misc = require 'rlntm_pkgs.misc'
local RLForce, parent = torch.class('nn.RLForce','nn.Module')

----- We shall rely on two global variables: 
----- global_action_seq, and
----- global_action_pos
----- we shall accumulate our probabilities in
----- global_action_probs

function RLForce:__init(action_map)
    parent.__init(self)
    self.action_map = action_map
end

function RLForce:updateOutput(input)
    assert(#input == 2)         
    local action_dist, baseline = unpack(input)
    local batch_size, num_possible_actions = unpack(action_dist:size():totable())

    self.probs = (self.probs or torch.zeros(action_dist:size())):resize(action_dist:size())
    self.probs:exp(action_dist)

    --- the action that we want to take is here: 
    self.sample = global_action_seq[global_action_pos]:clone() 
    global_action_pos = global_action_pos + 1

    assert (self.sample:max() <= num_possible_actions)
    assert (self.sample:min() >= 1)

    assert (global_action_probs:size(1) == batch_size and global_action_probs:nDimension() == 1)

    self.my_global_action_pos = global_action_pos

    --- collect probabilities of each trajectory: 

    for b = 1, batch_size do
        if DO_NOT_ACCUMULATE_PROBS ~= true then
            global_action_probs[b] = global_action_probs[b] * self.probs[{b, self.sample[b]}]        
        end
        
    end 
    --print ('self.probs = ', self.probs:sum(2))
    --print ('action_map = ', self.action_map)

    ---- apply action map.  Good.
    self.output:resize(self.sample:size())
    for b = 1, batch_size do
         self.output[b] = self.action_map[self.sample[b]]
    end
    return self.output
end


function RLForce:updateGradInput(input, gradOutput)
   local action_dist, baseline = unpack(input)
   local batch_size, num_possible_actions = unpack(action_dist:size():totable())
   if not misc.same_size(gradOutput, baseline) then
       if (gradOutput:nDimension() == 1 and baseline:nDimension() == 2 and gradOutput:nElement() == baseline:nElement() and baseline:size(2) == 1) then
           gradOutput = gradOutput:view(batch_size, 1)
       else
           assert (misc.same_size(gradOutput:size(), baseline:size()))
       end
   end
   local d_action_dist = torch.zeros(action_dist:size())
   local d_baseline = torch.add(gradOutput, -1, baseline)    ---  (future R - current baseline) 
   d_baseline:cmul(global_action_probs)  ---- everything is the same as in samples, except that we multiply the reward by the probabilities.
   for b = 1, batch_size do
       if math.abs(gradOutput[{b,1}]) > 1e-6 then
           d_action_dist[{b, self.sample[b]}] = d_baseline[{b,1}]
       else
           d_baseline[{b,1}] = 0
       end
   end 
   self.gradInput = {d_action_dist, d_baseline}
   return self.gradInput
end


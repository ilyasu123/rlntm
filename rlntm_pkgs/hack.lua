
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--

require 'nngraph'

local nesting = require 'rlntm_pkgs.nesting'
local utils = require 'rlntm_pkgs.utils'
local istensor = utils.istensor
local istable = utils.istable
local istorchclass = utils.istorchclass

local function getTotalGradOutput(node)
        local gradOutput = node.data.gradOutput
        assert(istable(gradOutput), "expecting gradients to sum")
        assert(gradOutput ~= nil) 
        if #gradOutput > 1 then
                node.data.gradOutputBuffer = node.data.gradOutputBuffer or nesting.cloneNested(gradOutput[1])
                local gobuff = node.data.gradOutputBuffer
                nesting.resizeNestedAs(gobuff, gradOutput[1])
                nesting.fillNested(gobuff, 0)
                for i=1,#gradOutput do
                        if gobuff == "nil" and gradOutput[i] ~= "nil" then
                            gobuff = nesting.cloneNested(gradOutput[i])
                        else
                            nesting.addNestedTo(gobuff, gradOutput[i])
                        end
                end
                gradOutput = gobuff
                assert (gradOutput ~= nil)
        else
                if (gradOutput[1] == nil) then
                    ---- uh oh: this is pretty bad!
                    print ("node.data.module = ", node.data.module)
                    ---- print ("node = ", node)
                    print ("#gradOutput = ", #gradOutput)
                    print ("so, gradOutput = ", gradOutput)
                    -- print ("node = ", node)
                end


                gradOutput = gradOutput[1]
                assert (gradOutput ~= nil)
        end
        return gradOutput
end


----- OK. So nil support is now present.  Let's test it.
function nn.gModule:updateGradInput(input,gradOutput)
        local function neteval(node)
                if node.data.selectindex then
                        assert(not node.data.module, "the selectindex-handling nodes should have no module")
                        assert(#node.children == 1, "only the splitted node should be the input")
                        local child = node.children[1]
                        local go = getTotalGradOutput(node)
                        child.data.gradOutput = child.data.gradOutput or {}
                        assert(#child.data.gradOutput <= 1, "the splitted node should be used only once")
                        -- The data.gradOutput holds the to-be-summed gradients.
                        child.data.gradOutput[1] = child.data.gradOutput[1] or {}
                        assert(not child.data.gradOutput[1][node.data.selectindex], "no gradOutput should be assigned yet")
                        child.data.gradOutput[1][node.data.selectindex] = go
                else
                        local gradOutput = getTotalGradOutput(node)
                        assert (gradOutput ~= nil)
                        -- updateGradInput through this node
                        -- If no module is present, the node behaves like nn.Identity.
                        local gradInput
                        if not node.data.module then
                                gradInput = gradOutput
                        else
                                local input = node.data.input
                                if #input == 1 then
                                        input = input[1]
                                end
                                local module = node.data.module
                                ------------- This must be one of woj's hacks, but I mostly understand what's going on! 
                                if istensor(gradOutput) then
                                     --assert(p == nil or p.gc == 1 or module.__typename == "nn.Identity" or gradOutput:type() == "torch.CudaTensor")
                                end
                                if istensor(input) then
                                     --assert(p == nil or p.gc == 1 or module.__typename == "FastEmbedding" or input:type() == "torch.CudaTensor")
                                end
                                -------------
                                if nesting.isNil(gradOutput) then
                                    gradInput = "nil"
                                else
                                    gradInput = module:updateGradInput(input,gradOutput)
                                end
                        end

                        -- propagate the output to children
                        for i,child in ipairs(node.children) do
                                child.data.gradOutput = child.data.gradOutput or {}
                                local mapindex = node.data.mapindex[child.data]
                                local gi

                                ---- extra code to make sure that nil turns to nil.
                                if gradInput ~= "nil" then
                                     ---- assert (#gradInput == #node.children)
                                     if #node.children == 1 then
                                           gi = gradInput
                                           assert (gi ~= nil)
                                     else
                                           gi = gradInput[mapindex]
                                           if (gi == nil) then
                                               print ("uh oh:")
                                               print ("gradInput = ", gradInput)
                                               print ("mapindex = ", mapindex)
                                               print ("#node.data.mapindex = ", #node.data.mapindex)
                                               print ("node.data.module = ", node.data.module)
                                               print ("#node.children = ", #node.children)
                                               print ("child.data.module = ", child.data.module)
                                           end
                                           assert (gi ~= nil)
                                     end
                                else
                                     gi = "nil"
                                end
                                assert (gi ~= nil)
                                table.insert(child.data.gradOutput, gi)
                        end
                end
                if self.verbose then
                        print(' V : ' .. node:label())
                end
        end
        local outnode = self.outnode
        if #outnode.children > 1 and #gradOutput ~= #outnode.children then
                error(string.format('Got %s gradOutputs instead of %s', #gradOutput, #outnode.children))
        end
        for _,node in ipairs(self.backwardnodes) do
                local gradOutput = node.data.gradOutput
                while gradOutput and #gradOutput >0 do
                        table.remove(gradOutput)
                end
        end
        -- Set the starting gradOutput.
        outnode.data.gradOutput = outnode.data.gradOutput or {}
        outnode.data.gradOutput[1] = gradOutput

        for i,node in ipairs(self.backwardnodes) do
    --if node.data.module then
    --  print(string.format("\t%d = %s", i, node.data.module.__typename))
    --end
                neteval(node)
        end

        assert(#self.innode.data.gradOutput == 1, "expecting the innode to be used only once")
        self.gradInput = self.innode.data.gradOutput[1]
        return self.gradInput
end

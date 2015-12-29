
--[[
  
RLNTM is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see http://creativecommons.org/licenses/by-nc/4.0/.

]]--


local nesting = {}

local utils = paths.dofile('utils.lua')
local istensor = utils.istensor

function nesting.isNil(obj)
       if obj == "nil" then
            return true
       elseif istensor(obj) then 
            return false 
       else
            for key, child in pairs(obj) do
                 --- if anything is not a nil, we are not nil
                 if not nesting.isNil(child) then
                       return false
                 end
            end
            return true       --- if everyone is a nil, then so are we.
       end
end

-- Creates a clone of a tensor or of a table with tensors.
function nesting.cloneNested(obj)
        if obj ==  "nil" then
                return "nil"
        end

	if istensor(obj) then
		return obj:clone()
	end

	local result = {}
	for key, child in pairs(obj) do
		result[key] = nesting.cloneNested(child)
	end
	return result
end

-- Fills the obj with the given value.
-- The obj can be a tensor or a table with tensors.
function nesting.fillNested(obj, value)
        if obj == "nil" then
              assert (value == 0, "nil can only be filled with zeros")
	elseif istensor(obj) then 
		obj:fill(value)
	else
		for key, child in pairs(obj) do
			nesting.fillNested(child, value)
		end
	end
end

-- Resizes all tensors in the output.
function nesting.resizeNestedAs(output, input)
        if output == "nil" then
            -- do nothing
        elseif input == "nil" then
            -- do nothing
	elseif istensor(output) then 
		output:resizeAs(input)
	else
		for key, child in pairs(input) do
			-- A new element is added to the output, if needed.
			if not output[key] then
				output[key] = nesting.cloneNested(child)
			else
				nesting.resizeNestedAs(output[key], child)
			end
		end
		-- Extra elements are removed from the output. 
		for key, child in pairs(output) do
			if not input[key] then
				output[key] = nil
			end
		end
	end
end

-- Adds the input to the output.
-- The input can contain nested tables.
-- The output will contain the same nesting of tables.
function nesting.addNestedTo(output, input)
        if input == "nil" then
            -- do nothing
        elseif output == "nil" then
            -- do nothing as well
	elseif istensor(output) then
             if input:dim() ~= output:dim() then
                 print("Output size", output:size())
                 print(output)
                 print("Input size", input:size())
                 print(input)
                 print("_______________")
             end
	     output:add(input)
	else
             for key, child in pairs(input) do
		assert(output[key] ~= nil, "missing key")
                ---- main modification:  allow bprop to send "nil"s back.
                if output[key] == "nil" then
                    if child ~= "nil" then
                        output[key] = nesting.cloneNested(child)
                    end
                elseif child == "nil" then
                    --- do nothing
                else
                    nesting.addNestedTo(output[key], child)
                end
             end
	end
end

return nesting

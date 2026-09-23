local M = {}

function M.run(cmd)
  return os.execute(cmd)
end

local function helper(x)
  return x + 1
end

return M

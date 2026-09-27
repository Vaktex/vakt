require "open3"

DEBUG = true

def run(cmd)
  system(cmd)
end

class Store
  def get(key)
    @data[key]
  end
end

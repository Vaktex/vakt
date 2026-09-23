object Main {
  def run(cmd: String): Int = sys.process.Process(cmd).!
}

class Store {
  def get(k: String): String = k
}

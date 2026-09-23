using System.Data.SqlClient;

namespace Demo {
    public class Repo {
        public Repo() { }

        public object Find(string id) {
            var cmd = new SqlCommand("SELECT * FROM t WHERE id=" + id);
            return cmd.ExecuteScalar();
        }
    }
}

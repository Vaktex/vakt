package demo;

import java.sql.*;

public class Sample {
    private Connection conn;

    public Sample(Connection c) {
        this.conn = c;
    }

    // Finds a user.
    public ResultSet find(String name) throws SQLException {
        return conn.createStatement().executeQuery("SELECT * FROM u WHERE n='" + name + "'");
    }
}

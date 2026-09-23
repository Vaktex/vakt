import express from "express";

const app = express();
app.use(express.json());
app.listen(3000);

// Look up a user by id.
export async function getUser(id: string): Promise<User> {
  return db.query(`SELECT * FROM users WHERE id = ${id}`);
}

export const verify = (token: string): boolean => token.length > 0;

class UserService {
  private cache = new Map<string, User>();
  find(id: string) {
    return this.cache.get(id);
  }
}

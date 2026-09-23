import java.io.File

fun read(path: String): String {
    return File(path).readText()
}

class Repo {
    fun find(id: Int): String = "x$id"
}

<?php
$db = new PDO("sqlite::memory:");
$x = 1;
$y = 2;

function find($id) {
    global $db;
    return $db->query("SELECT * FROM t WHERE id = " . $id);
}

class Ctl {
    public function show($req) {
        echo $req["q"];
    }
}

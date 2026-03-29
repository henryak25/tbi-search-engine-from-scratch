class TrieNode:
    """
    Node untuk Trie standar.
    Setiap node hanya menyimpan satu karakter yang dipetakan ke child node berikutnya.
    """
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.value = None   # Valuenya Term ID atau Doc ID


class Trie:
    """
    Implementasi Prefix Tree (Trie).
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, value):
        """Insert kata ke dalam Trie karakter demi karakter."""
        node = self.root
        for char in word:
            # Jika karakter belum ada di cabang saat ini, buat node baru
            if char not in node.children:
                node.children[char] = TrieNode()
            # Pindah ke node anak
            node = node.children[char]
        
        # Tandai akhir kata dan simpan ID-nya
        node.is_end_of_word = True
        node.value = value

    def search(self, word):
        """Mengembalikan Value (ID) jika ada, atau None."""
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        
        if node.is_end_of_word:
            return node.value
        return None

    def search_prefix(self, prefix):
        """Mengembalikan list berisi semua kata yang berawalan 'prefix'"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return [] # Prefix tidak ditemukan sama sekali
            node = node.children[char]
        
        # Lakukan DFS (Depth First Search) untuk mengumpulkan semua sisa kata
        results = []
        def dfs(current_node, current_word):
            if current_node.is_end_of_word:
                results.append(current_word)
            for char, child_node in current_node.children.items():
                dfs(child_node, current_word + char)
        
        dfs(node, prefix)
        return results

    # Magic Methods agar Trie bisa diperlakukan mirip seperti Dictionary Python biasa
    def __contains__(self, word):
        return self.search(word) is not None

    def __getitem__(self, word):
        res = self.search(word)
        if res is None:
            raise KeyError(word)
        return res

    def __setitem__(self, word, value):
        self.insert(word, value)

class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self, use_trie=False):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.use_trie = use_trie
        self.str_to_id = Trie() if use_trie else {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        return len(self.id_to_str)

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        return self.id_to_str[i]

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        if s not in self.str_to_id:
            self.id_to_str.append(s)
            self.str_to_id[s] = len(self.id_to_str) - 1
        return self.str_to_id[s]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError

def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparablem, int)]
        Penggabungan yang sudah terurut
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge

if __name__ == '__main__':
    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]

    # Validate IdMap behavior with both dict and Trie
    for use_trie in [False, True]:
        term_id_map = IdMap(use_trie=use_trie)
        assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
        assert term_id_map[1] == "semua", "term_id salah"
        assert term_id_map[0] == "halo", "term_id salah"
        assert term_id_map["selamat"] == 2, "term_id salah"
        assert term_id_map["pagi"] == 3, "term_id salah"

        doc_id_map = IdMap(use_trie=use_trie)
        assert [doc_id_map[docname] for docname in docs] == [0, 1, 2], "docs_id salah"

    # Direct Trie sanity checks
    trie = Trie()
    trie.insert("alpha", 7)
    trie.insert("beta", 9)
    assert trie.search("alpha") == 7, "trie search alpha salah"
    assert trie.search("beta") == 9, "trie search beta salah"
    assert trie.search("alp") is None, "trie prefix bukan full word harus None"
    assert ("alpha" in trie) is True, "trie contains alpha salah"
    assert ("gamma" in trie) is False, "trie contains gamma salah"

    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)], \
                                      [(1, 11), (2, 4), (4, 3 ), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "sorted_merge_posts_and_tfs salah"

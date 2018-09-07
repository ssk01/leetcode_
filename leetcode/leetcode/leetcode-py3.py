from collections import deque
import string
import collections
import json
from queue import PriorityQueue
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

def stringToIntegerList(input):
    return json.loads(input)

def stringToListNode(input):
    # Generate list from the input
    numbers = stringToIntegerList(input)

    # Now convert that list into linked list
    dummyRoot = ListNode(0)
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next
    return ptr
class Trie:
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dicts = {}
        self.end = False

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        if not word:
            self.end = True
        else:
            c = word[0]
            if c not in self.dicts:
                self.dicts[c] = Trie()
            self.dicts[c].insert(word[1:])

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        if not word:
            return self.end
        c = word[0]
        if c not in self.dicts:
            return False
        else:
            return self.dicts[c].search(word[1:])

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        if not prefix:
            return self.end or (len(self.dicts) != 0)
        c = prefix[0]
        if c in self.dicts:
            return self.dicts[c].startsWith(prefix[1:])
        else:
            return False
class Solution(object):
    def isPalindrome1(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow, fast = head, head
        fastPrev = None
        while fast and fast.next:
            slow = slow.next
            fastPrev = fast.next
            fast = fast.next.next
        end = fast
        if not fast:
            end = fastPrev
        def rev(b, e):
            # if b == e: return e
            # cur = b.next
            cur = b
            prev = None
            while cur != e:
                nex = cur.next
                cur.next = prev
                prev = cur 
                cur = nex
            cur.next = prev
            return e
        nHead= rev(slow, end)
        result = True
        while nHead and head:
            if nHead.val == head.val:
                nHead = nHead.next
                head = head.next
            else:
                result = False
                break
        rev(end, slow)
        return result
    def validPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        i = 0 
        j = len(s)-1
        def isP(m, n):
            a = s[m:n+1]
            return  a[::-1] == a
        
        while s[i] == s[j] and i < j:
            i+=1
            j-=1
        if i>=j:
            return True
        
        return isP(i+1, j) or isP(i, j-1)
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s == s[::-1]:
            return len(s)

        n = len(s)
        dp = [0 for j in range(n)]
        newdp = [1 for j in range(n)]
        dp[n-1] = 1

        for i in range(n-1, -1, -1):   # can actually start with n-2...
            #newdp = dp[:]
            for k in range(i+1, n):
                newdp[k]=dp[k]
            #newdp[i] = 1
            for j in range(i+1, n):
                if s[i] == s[j]:
                    newdp[j] = 2 + dp[j-1]
                else:
                    newdp[j] = max(dp[j], newdp[j-1])
            for k in range(i, n):
                dp[k] = newdp[k]
                    
        return dp[n-1]
    def palindromePairs(self, words):
        """
        :type words: List[str]
        :rtype: List[List[int]]
        """
        maps = dict([(w[::-1], i) for i, w in enumerate(words)])
        res = []
        for j, w in enumerate(words):
            for i in range(len(w)):
                pre = w[:i]
                post = w[i:]
                if pre in maps and maps[pre] != j and post == post[::-1]:
                    res.append([j, maps[pre]])
                if post in maps and maps[post] != j and pre == pre[::-1]:
                    res.append([maps[post], j])
        return res
    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        cut = [1086 for x in range(len(s))]
        for i in range(len(s)):
            # odd
            j = 0
            while i-j>=0 and i+j<len(s) and s[i+j] == s[i-j]:
                if i-j == 0:
                    cut[i+j] = 0
                else:
                    cut[i+j] = min(cut[i+j], cut[i-j-1]+1)
                j+=1
            # even
            # if i+1 < s[i] == s[i+1]:
            j = 0
            while i-j>=0 and i+j+1<len(s) and s[i-j] == s[i+j+1]:
                if i-j == 0:
                    cut[i+j+1] = 0
                else:
                    cut[i+j+1] = min(cut[i+j+1], cut[i-j-1]+1)
                j+=1

        return cut[len(s)-1]

    def isp(self, s, i, j):
        while j>i:
            if s[j] ==s[i]:
                j-=1
                i+=1
            else:
                return 0
        return 1
        # return  min(res)-1
    # def minCut1(self, s):
    #         """
    #     :type s: str
    #     :rtype: int
    #     """
    #     res =[]
    #     lens = len(s)
    #     def fuckhel( idx, tmp):
 
    #         if idx == lens: 
    #             res.append(len(tmp))
    #             return
    #         for i in range(idx, lens):
    #             if idx > 0:
    #                 if s[idx:i+1] == s[i:idx-1:-1]: 
    #                     tmp.append(1)
    #                     fuckhel(i+1,tmp)
    #                     tmp.pop()
    #             elif idx == 0 and s[idx:i+1] == s[i::-1]: 
    #                 tmp.append(1)
    #                 fuckhel(i+1,tmp)
    #                 tmp.pop()
    #     fuckhel(0,[])
    #     return  min(res)-1
    def shortestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        i = 0
        # j = len(s1)-1
        end = 0
        for j in range(len(s)-1, -1,-1):
            i = 0
            while j > i:
                if s[i] == s[j]:
                    i+=1
                    j-=1
                else:
                    break
            if j==i:
                end = 2*i
                break
            if j == i-1:
                end = 2*j+1
                break
        return s[end+1:][::-1] + s
    def lengthOfLIS(self, nums):
        tails = [0]*len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i < j:
                mid = (i+j) // 2
                if x > tails[mid]:
                    i = mid + 1
                else:
                    j = mid
            tails[i] = x
            size = max(i + 1, size)
        return size 


    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        dummyNode = ListNode(0)
        curr = dummyNode
        q = PriorityQueue()
        for node in lists:
            if node:
                q.put((node.val, node))
        while q.qsize() > 0:
            curr.next = q.get()[1]
            curr = curr.next
            if curr.next:
                q.put((curr.next.val, curr.next))
        return dummyNode.next

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        res = 0
        stk = [-1]
        for i in range(len(s)):
            if s[i] == '(':
                stk.append(i)
            else:
                if stk[-1] != -1 and s[stk[-1]] == '(':
                    stk.pop()
                    res = max(res, i - stk[-1])
                else:
                    stk.append(i)
        
        # stk = []
        # for i in range(len(s)):
        #     if s[i] == '(':
        #         stk.append(i)
        #     else:
        #         if stk:
        #             if s[stk[-1]] == '(':
        #                 stk.pop()
        #             else:
        #                 stk.append(i)
        #         else:
        #             stk.append(i)
        # if not stk:
        #     return len(s)
        # else:
        #     res = 0
        #     end = len(s)
        #     beg = 0
        #     while stk:
        #         beg = stk[-1]
        #         stk.pop()
        #         res = max(res, end - beg - 1)
        #         end = beg
        #     res = max(res, beg)
        #     return res
            

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        # stk = []
        # maps={'(':')','{':'}','[':']'}
        # i = 0
        # res = 0
        # tres = 0
        # for chr in s:
        #     print(chr)
        #     if chr == '(' or chr == '[' or chr == '{':
        #         stk.append(chr)
        #     else:
        #         if stk:
        #             tmp = stk.pop()
        #             if chr != maps[tmp]:
        #                 i = 0
        #                 tres = 0
        #             else:
        #                 i += 1
        #             if not stk:
        #                 tres= i
        #                 res = max(res, tres)
        #         else:
        #             i = 0
        #             tres = 0
        # if stk:
        #     if tres == 0:
        #         res = max(res, i)
            
        # return res*2
        
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        visit = [0 for _ in range(numCourses)]
        for x, y in prerequisites:
            graph[y].append(x)
        def dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            visit[i] = -1
            for j in graph[i]:
                if not dfs(j):
                    return False
            visit[i] = 1
            return True
        for i in range(numCourses):
            if not dfs(i):
                return False
        return True

        # tu = {}
        # for i in range(len(prerequisites)):
        #     if prerequisites[i][1] not in tu:
        #         tu[prerequisites[i][1]] = []
        #     tu[prerequisites[i][1]].append(prerequisites[i][0])
        # dus = [0] * numCourses
        # for k in tu:
        #     for val in tu[k]:
        #         dus[val] += 1
            
        # print(dus)
        # for i in range(numCourses):
        #     index = numCourses            
        #     for j in range(numCourses):
        #         if dus[j] == 0:
        #             index = j
        #             break
        #     if index == numCourses:
        #         return False
        #     dus[j] = -1
        #     if index in tu:
        #         for nexts in tu[index]:
        #             dus[nexts]-=1
        # return True

    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        sl = len(s)
        tl = len(t)
        if sl != tl: return False
        maps = {}
        mapp = {}
        for i in range(sl):
            if s[i] not in maps:
                if t[i] not in mapp:
                    maps[s[i]] = t[i]
                    mapp[t[i]] = 1
                else:
                    return False
            else:
                if maps[s[i]] == t[i]:
                    pass
                else:
                    return False
        return True

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        def notinSquare(i, j, c):
            i = i - i % 3
            j = j - j % 3
            for m in range(3):
                for n in range(3):
                    if board[i+m][j+n] == c:
                        return False
            return True
        def notinCol(j, c):
            for i in range(len(board)):
                if board[i][j] == c:
                    return False
            return True
        def notinRow(i, c):
            for chr in board[i]:
                if chr == c:
                    return False
            return True
        nums = '123456789'
        def help(pos):
            if pos == 81: return True
            i = pos // 9
            j = pos % 9
            if board[i][j] != '.':
                return help(pos+1)
            else:
                for ch in nums:
                    if notinCol(j, ch) and notinRow(i, ch) and notinSquare(i, j, ch):
                        tmp = board[i]
                        board[i] = tmp[:j] + ch + tmp[j+1:]
                        if (help(pos+1)):
                            return True
                        board[i] = tmp
                return False
        help(0)

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        need, missing = collections.Counter(t), len(t)
        i = I = J = 0
        for j, c in enumerate(s):
            if need[c] > 0:
                missing -= 1
            need[c] -= 1
            if not missing:
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1
                if not J or j-i <= J-I:
                    I, J = i, j
        return s[I:J+1]  







    def findSubstring(self, s, words):
        """
        :type s: str
        :type words: List[str]
        :rtype: List[int]
        """
        count = {}
        for word in words:
            if word not in count:
                count[word] = 0
            count[word] += 1
        wordLens = len(words[0])
        wordcount = len(words)
        strLens = wordLens * wordcount
        res = []
        slens = len(s)
        print(count)
        for i in range(wordLens):
            seen = {}
            j = 0
            m = 0
            while j < wordcount and i+(j+m+1)*wordLens <= slens:
                w = s[i+(j+m)*wordLens:i+(j+m+1)*wordLens]
                if w in count:
                    if w not in seen:
                        seen[w] = []
                    seen[w].append(j+m)
                    if len(seen[w]) > count[w]:
                        jm = seen[w][0]
                        seen[w] = seen[w][1:]
                        m += 1
                        j = 0
                        continue
                    if j == wordcount - 1:
                        res.append(i+m*wordLens)
                        seen[s[i+m*wordLens:i+(m+1)*wordLens]] -= 1
                        m += 1
                        continue                       
                else:
                    m = j+m+1
                    j = 0
                    seen = {}
                    continue
                j += 1
        return res
        # for i in range(len(s)-strLens+1):
        #     seen = {}
        #     j = 0
        #     while j < wordcount:
        #         w = s[i+j*wordLens:i+(j+1)*wordLens]
        #         if w in count:
        #             if w not in seen:
        #                 seen[w] = 0
        #             seen[w] += 1
        #             if seen[w] > count[w]:
        #                 break
        #         else:
        #             break
        #         j += 1
        #     if j == wordcount:
        #         res.append(i)
        # return res
            

    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        maps  = {}
        maps[None]= None
        pre = dummyNode = RandomListNode(0)
        t = head
        h = head
        while h:
            tmp = RandomListNode(h.label)
            maps[h] = tmp
            pre.next = tmp
            pre = tmp
            h = h.next
        df = dummyNode.next
        while t:
            df.random = maps[t.random]
            t = t.next
            df = df.next
        return dummyNode.next
        
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = head
        slow = head
        entry = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                while slow != entry:
                    slow = slow.next
                    entry = entry.next
                return entry
        return None                

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        end = len(s) - 1
        beg = 0
        while beg < end:
            while beg < end and not s[beg].isalnum():
                beg += 1
            while beg < end and not s[end].isalnum():
                end -= 1
            if beg < end:
                if s[beg].lower() == s[end].lower():
                    beg += 1
                    end -= 1
                else:
                    return False
            else:
                return True

        return True
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head: return None
        lens = 0
        head1= head
        while head1 and head1.next:
            head1 = head1.next
            lens += 1
        lens += 1
        head1.next = head
        k = lens - k % lens
        if k == lens:
            k = 0
        
        while k > 0:
            head1 = head1.next
            k -= 1
        res = head1.next
        head1.next = None
        return res  

        

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if not matrix or not matrix[0]: return 0
        n = len(matrix[0])
        height = [0]* (n+1)
        ans = 0

        for row in matrix:
            for i in xrange(n):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            
            stack = [-1]
            for i in xrange(n+1):
                while height[i] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = i - 1 - stack[-1]
                    ans = max(ans, w * h)
                stack.append(i)
        return ans 

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        if not heights: return 0
        left = self.left(heights)
        right = self.right(heights)
        res = 0
        tmp = 0
        for i in range(len(heights)):
            tmp = heights[i]*(left[i] + right[i] + 1)
            res = max(res, tmp)
        return res

    def left(self, heights):
        distance = [0]*len(heights)
        for i in range(len(heights)):
            j = i -1
            t = 0
            tmp = heights[i]
            while j >= 0:
                if tmp <= heights[j]:
                    t = t + distance[j] + 1
                    j = j - distance[j] - 1
                else:
                    break
            distance[i] = t
        return distance
    def right(self,heights):
        distance = [0]*len(heights)
        for i in range(len(heights)-1,-1,-1):
            t = 0
            j = i+1
            tmp = heights[i]
            while j < len(heights):
                if tmp <= heights[j]:
                    t = t + distance[j] + 1
                    j = j + distance[j] + 1
                else: 
                    break     
            distance[i] = t
        return distance
            
                

    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        #left to right
        # before one use up . the small one
        if not s: return ''
        dicts = {}
        visted = {}
        result = "0"

        for c in 'abcdefghijklmnopqrstuvwxyz':
            dicts[c] = 0
            visted[c] = False
        for c in s:
            dicts[c]+=1

        for c in s:
            dicts[c]-=1
            if (visted[c]):
                continue
            while c < result[-1] and dicts[result[-1]]:
                visted[result[-1]] = False
                result=result[:-1]
            result += c
            visted[c] = True
        return result[1:]
        
        

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        lens = len(s1)
        if lens<=3 and s1 == s2:
            return True        
        count ={}
        for i in range(lens):
            if s1[i] not in count:
                count[s1[i]] = 0
            count[s1[i]] += 1
            if s2[i] not in count:
                count[s2[i]] = 0
            count[s2[i]] -= 1
        for val in count.values():
            if val != 0:
                return False
        for i in range(1,lens):
            if self.isScramble(s1[:i], s2[:i]) and self.isScramble(s1[i:], s2[i:]):
                return True
            if self.isScramble(s1[:i], s2[lens-i:]) and self.isScramble(s1[i:], s2[:lens-i]):
                return True
        return False







    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """
        if endWord not in wordList: return []
        wordlens = len(beginWord)
        def compare(i, j):
            count = 0
            for k in range(wordlens):
                if wordList[i][k] != wordList[j][k]:
                    count += 1
                    if count > 1:
                        break
            if count == 1: return True
            return False

        maps={}
        endPos = wordList.index(endWord)
        wordList.append(beginWord)
        begPos = len(wordList) - 1
        visted = [False] * len(wordList)
        visted[begPos] = True
        dicts = {}
        for i in range(len(wordList)):
            dicts[wordList[i]] = i
        queue = deque([(begPos, 1)])
        levelVisting = set([begPos])
        endLevels = 100
        nowLevels = 0
        l = 0
        while queue:
            pos, levels = queue.popleft()
            if pos == endPos:
                endLevels = levels

            if levels > endLevels:
                break

            if l != levels:
                for p in levelVisting:
                    visted[p] = True
                levelVisting = set()
                l = levels

            for k in range(wordlens):
                for c in string.ascii_lowercase :
                    next_words = wordList[pos][:k] + c + wordList[pos][k+1:]
                    if next_words in dicts:
                        i = dicts[next_words]
                        if not visted[i]:
                            queue.append((i, levels+1))
                            if i not in maps: maps[i] =set()
                            maps[i].add(pos)
                            levelVisting.add(i)
        #             print("1",maps)
        #     print("       ",queue)        
        # print("1",maps)

        if endPos not in maps: return []
        res=[]
        def generatePath(i, Path):
            if i == begPos:
                res.append([wordList[p] for p in Path])
                return
            for p in maps[i]:
                generatePath(p,[p]+Path)
        generatePath(endPos, [endPos])
        return res

        # if endWord not in wordList: return []
        # wordlens = len(beginWord)
        # def compare(i, j):
        #     count = 0
        #     for k in range(wordlens):
        #         if wordList[i][k] != wordList[j][k]:
        #             count += 1
        #             if count > 1:
        #                 break
        #     if count == 1: return True
        #     return False

        # maps={}
        # endPos = wordList.index(endWord)
        # wordList.append(beginWord)
        # begPos = len(wordList) - 1
        # print(wordList)
        # visted = [False] * len(wordList)
        # visted[begPos] = True

        # queue = deque([(begPos, 1)])
        # levelVisting = set([begPos])
        # endLevels = 100
        # nowLevels = 0
        # l = 0
        # while queue:
        #     pos, levels = queue.popleft()
        #     if pos == 4:
        #         aa=1
        #     if pos == endPos:
        #         endLevels = levels

        #     if levels > endLevels:
        #         break

        #     if l != levels:
        #         for p in levelVisting:
        #             visted[p] = True
        #         levelVisting = set()
        #         l = levels
        #     for i in range(len(wordList)):
        #         if not visted[i] and compare(i, pos):
        #             queue.append((i, levels+1))
        #             if i not in maps: maps[i] =set()
        #             maps[i].add(pos)
        #             levelVisting.add(i)
        #             print("1",maps)
        #     print("       ",queue)        
        # print("1",maps)

        # if endPos not in maps: return []
        # res=[]
        # def generatePath(i, Path):
        #     if i == begPos:
        #         res.append([wordList[p] for p in Path])
        #         return
        #     for p in maps[i]:
        #         generatePath(p,[p]+Path)
        # generatePath(endPos, [endPos])
        # return res
            
        # while queue:
        #     # print(queue)
        #     word, steps, paths = queue.popleft()
        #     if word == endWord:
        #         res.append(paths)
        #         lens = steps
        #         break
        #     if l != lens:
        #         for vals in maps:
        #             visted.add(vals)
        #         l = lens
        #     if word not in visted:
        #         next_words = dict_words.get(word, [])
        #         for nexts in next_words:
        #             if nexts != word and nexts not in visted:
        #                 queue.append((nexts, steps+1, paths+[nexts]))
        #                 tmp = maps.setdefault(steps,set())
        #                 tmp.add(nexts)


        # def construct_dict(word_list):
        #     def compare(lhs, rhs):
        #         k = 0
        #         for i in range(len(lhs)):
        #             if lhs[i] != rhs[i]:
        #                 k+=1
        #             if k > 1:
        #                 return -1
        #         return k
        #     d = {}
            
        #     for lhs in word_list:
        #         for rhs in word_list:
        #             if compare(lhs,rhs) == 1:
        #                 tmp = d.setdefault(lhs, set())
        #                 tmp.add(rhs)
        #                 tmp = d.setdefault(rhs, set())
        #                 tmp.add(lhs)
                        
        #         # for i in range(len(word)):
        #         #     s = word[:i]+"_"+word[i+1:]
        #         #     d[s] = d.get(s, []) + [word]
        #     return d
        # lens = 0
        # res = []
        # if endWord not in wordList: return []
        # if beginWord not in wordList: wordList.append(beginWord)

        # dict_words = construct_dict(wordList)

        # queue, visted = deque([(beginWord, 1,[beginWord])]), set()
        # maps ={1:set([beginWord])}

        # l = 0
        # while queue:
        #     # print(queue)
        #     word, steps, paths = queue.popleft()
        #     if word == endWord:
        #         res.append(paths)
        #         lens = steps
        #         break
        #     if l != lens:
        #         for vals in maps:
        #             visted.add(vals)
        #         l = lens
        #     if word not in visted:
        #         next_words = dict_words.get(word, [])
        #         for nexts in next_words:
        #             if nexts != word and nexts not in visted:
        #                 queue.append((nexts, steps+1, paths+[nexts]))
        #                 tmp = maps.setdefault(steps,set())
        #                 tmp.add(nexts)
        # while queue:
        #     word, steps, paths = queue.popleft()
        #     if steps != lens:
        #         break
        #     if word == endWord:
        #         res.append(paths)
        # return res

        #tle
        # def construct_dict(word_list):
        #     d = {}
        #     for word in word_list:
        #         for i in range(len(word)):
        #             s = word[:i]+"_"+word[i+1:]
        #             d[s] = d.get(s, []) + [word]
        #     return d
        # lens = 0
        # res = []
        # if endWord not in wordList: return 0
        # if beginWord not in wordList: wordList.append(beginWord)

        # dict_words = construct_dict(wordList)

        # queue, visted = deque([(beginWord, 1,[beginWord])]), set()
        # maps ={1:set([beginWord])}

        # l = 0
        # while queue:
        #     word, steps, paths = queue.popleft()
        #     if word == endWord:
        #         res.append(paths)
        #         lens = steps
        #         break
        #     if l != lens:
        #         for vals in maps.values():
        #             visted.add(vals)
        #         l = lens
        #     if word not in visted:
        #         for i in range(len(word)):
        #             s = word[:i]+"_"+word[i+1:]
        #             next_words = dict_words.get(s, [])
        #             for nexts in next_words:
        #                 if nexts != word and nexts not in visted:
        #                     queue.append((nexts, steps+1, paths+[nexts]))
        #                     tmp = maps.setdefault(steps,set())
        #                     tmp.add(nexts)
        # while queue:
        #     word, steps, paths = queue.popleft()
        #     if steps != lens:
        #         break
        #     if word == endWord:
        #         res.append(paths)
        # return res



        #TTTTTLLLLLEEEEE
        # if endWord not in wordList: return []
        # queue = deque([[beginWord,1, [beginWord]]])
        # res = []
        # pathLens = -1
        # uppercase = "abcdefghijklmnopqrstuvwxyz"
        # dicts = set(wordList)
        # maps={}
        # l = 1
        # while queue:
        #     word, lens, paths= queue.popleft()
        #     if l != lens:
        #         for val in maps[l]:
        #             dicts.remove(val)
        #         l = lens
        #     if word == endWord:
        #         res.append(paths)
        #         pathLens = len(paths)
        #         break
        #     for i in range(len(word)):
        #         for c in uppercase:
        #             next_words = word[:i] + c + word[i+1:]
        #             # if next_words == endWord:
        #                 # queue.append([next_words,l+1,paths+[next_words]])
        #             if next_words in dicts:
        #                 tmp = maps.setdefault(l,set())
        #                 tmp.add(next_words)
        #                 queue.append([next_words,l+1,paths+[next_words]])
        # while queue:
        #     word, lens,paths = queue.popleft()
        #     if pathLens == lens:
        #         if word == endWord:
        #             res.append(paths)
        #     else:
        #         break
        # return res



    # def ladderLength(self, beginWord, endWord, wordList):
    #     if endWord not in wordList: return 0
    #     queue = deque([(beginWord, 1)])
    #     uppercase = "abcdefghijklmnopqrstuvwxyz"
    #     dicts = set(wordList)
    #     while queue:
    #         word, steps = queue.popleft()
    #         if word == endWord:
    #             return steps
    #         for i in range(len(word)):
    #             for c in uppercase:
    #                 next_words = word[:i] + c + word[i+1:]
    #                 if next_words in dicts:
    #                     dicts.remove(next_words)
    #                     queue.append((next_words, steps+1))
    #     return 0


    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        def construct_dict(word_list):
            d = {}
            for word in word_list:
                for i in range(len(word)):
                    s = word[:i]+"_"+word[i+1:]
                    d[s] = d.get(s, []) + [word]
            return d
        def bfs_words(beg, end, dict_words):
            queue, visted = deque([(beg, 1)]), set()
            while queue:
                word, steps = queue.popleft()
                if word not in visted:
                    visted.add(word)
                    if word == end:
                        return steps
                    for i in range(len(word)):
                        s = word[:i]+"_"+word[i+1:]
                        next_words = dict_words.get(s, [])
                        for nexts in next_words:
                            if nexts not in visted:
                                queue.append((nexts, steps+1))
            return 0
        if endWord not in wordList: return 0
        if beginWord not in wordList: wordList.append(beginWord)
        dicts = construct_dict(wordList)
        return bfs_words(beginWord, endWord, dicts)

    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 3:
            return 0
        primes = [0, 1]*(n/2)

        if n % 2 != 0:
            primes.append(0)
        primes[1] = 0
        primes[2] = 1
        for i in range(2,int(n**0.5)+1):
            if primes[i]:
                for j in range(i*i, n, i):
                    primes[j] = 0
        return sum(primes)
            
            
    # def countPrimes(self, n):
    #     """
    #     :type n: int
    #     :rtype: int
    #     """
    #     if n < 7:
    #         if n == 3: return 1
    #         if n == 1 or n == 0 or n == 2: return 0
    #         if n == 6: return 3
    #         return 2
    #     primes = [2,3,5]
    #     isPrimes = True
    #     for i in range(7,n,2):
    #         isPrimes = True
    #         half = i**0.5
    #         for num in primes:
    #             if num <= half and i % num == 0:
    #                 isPrimes = False
    #                 break
    #         if isPrimes == True:
    #             primes.append(i)
    #     return len(primes)    

    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        f1 =[x for x in version1.split('.')]       
        f2 =[x for x in version2.split('.')]
        i = 0
        while i < len(f1) and i <len(f2):
            if int(f1[i]) > int(f2[i]):
                return 1
            if int(f1[i]) < int(f2[i]):
                return -1
            i+=1
        while i < len(f1):
            if int(f1[i]) > 0:
                return 1
            i+=1
        while i < len(f2):
            if int(f2[i]) > 0:
                return -1
            i+=1
        return 0

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def confirm(i, j):
            if j == 3:
                if s[i] not in '12': return False
                if s[i] in '2':
                    if s[i+1] >'5':
                        return False
                    if s[i+1] == '5' and s[i+2] >'5':
                        return False
                    
            elif j == 2:
                return s[i] != '0'
            return True
        lens = len(s)
        res = []
        for i in range(1,4):
            if  lens -i <= 9 and lens - i >= 3 and confirm(0,i):
                for j in range(1,4):
                    if lens - i -j <= 6 and lens -i -j >= 2 and confirm(i, j):
                        for k in range(1,4):
                            if lens - i - j - k <= 3 and  lens -i -j - k >= 1 and confirm(i+j, k):
                                m = lens - i - j - k
                                if confirm(i+j+k, m):
                                    res.append(s[0:i]+"."+s[i:j+i]+"."+s[j+i:j+i+k]+"."+s[j+i+k:lens])
        return res
    
    
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        lists = [x for x in s.split(' ') if x != '']
        if len(lists) == 0: return ""
        if len(lists) == 1: return lists[0]
        else:
            lists = [lists[-1]]+lists[:len(lists)-1]
            return " ".join(lists)
        
                    
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid: return 0
        res = 0
        rows = len(grid)
        cols = len(grid[0])
        def fuck(i,j):
            if i>= 0 and i <rows and j>=0 and j < cols and grid[i][j] == 1:
                grid[i][j] = 0
                fuck(i, j-1)
                fuck(i, j+1)
                fuck(i+1, j)
                fuck(i-1, j)
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    fuck(i,j)
                    res+=1
        return res
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        maps = {}
        for i in range(len(list1)):
            maps[list1[i]] = i
        idxSum = len(list1)+len(list2)
        idx = []
        for j in range(len(list2)):
            if list2[j] in maps:
                maps[list2[j]] +=j
                if idxSum > maps[list2[j]]:
                    idx=[]
                    idx.append(j)
                    idxSum = maps[list2[j]]
                elif idxSum == maps[list2[j]]:
                    idx.append(j)
        res = []
        for i in idx:
            res.append(list2[i])
        return res
            




    def largestNumber(self, nums):
        # @param {integer[]} nums
        # @return {string}
        num = [str(x) for x in nums]
        num.sort(cmp=lambda x, y: cmp(y+x,x+y))
        return "0" if num and num[0] == "0" else "".join(num)




    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        pre = -2**32
        nums.append(pre)
        for i in range(len(nums)-1):
            if nums[i]>pre and nums[i]>nums[i+1]:
                return i
            pre = nums[i]

            
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = nums[0]
        maxs = res
        mins = res
        for i in range(1,len(nums)):
            if nums[i] < 0:
                maxs, mins = mins, maxs
            maxs = max(nums[i],nums[i]*maxs)
            mins = min(nums[i],nums[i]*mins)
            res = max(res, maxs)
        return res

    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<2:return 0
        nums.sort()
        res = 0
        pre = nums[0]
        for num in nums:
            res = max(num-pre,res)
            pre = num
        return res
        
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        beg = 0
        end = len(nums)-1
        while beg < end:
            if nums[beg]<nums[end]:
                return nums[beg]
            else:
                mid = (beg+end)//2
                if nums[mid] > nums[end]:
                    beg = mid + 1
                elif nums[mid] < nums[end]:
                    end = mid
                else:
                    end-=1
        return nums[end]
                    









    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def fuck(i,j):
            if j-i==0: return nums[i]
            if j-i==1: return min(nums[i],nums[j])
            mid = (i+j)//2
            if nums[mid] < nums[i]:
                return fuck(i,mid)
            else:
                return min(fuck(i,mid),fuck(mid+1,j))
        res = fuck(0,len(nums)-1)
        return res
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """    
        mask = 0x11 | (0x11<<8)
        mask = mask | (mask<<16)
        sums = n & mask
        sums += (n>>1) & mask
        sums += (n>>2) & mask
        sums += (n>>3) & mask
        mask = 0XFFFF
        sums = (sums >> 16) + (sums & mask)
        mask = 0XF0F
        sums = ((sums>>4)&mask) + (sum & mask)
        mask =0xff
        return (sums&mask)+(sums>>8)
    def reverseBits(self, n):
        # @param n, an integer
        # @return an integer
        m = 0 
        for i in range(32):
            m <<= 1
            m |= n&1
            n>>=1
        return m


    def findWords(self, board, words):
        trie = {}
        for w in words:
            t = trie
            for c in w:
                if c not in t:
                    t[c] = {}
                t = t[c]
            t['#'] = w
        res = []
        rows = len(board)
        cols = len(board[0])
        boolarray = [[True for _ in range(cols)] for _ in range(rows)]
        def dfs(i, j , tries):
            if not boolarray[i][j]:
                return
            if board[i][j] not in tries:
                return
            boolarray[i][j] = False
            tmp = board[i][j]            
            next_tries = tries[tmp]
            if '#' in next_tries and next_tries['#'] != '#':
                res.append(next_tries['#'])
                next_tries['#'] = '#'
            if i - 1 >= 0:  dfs(i-1, j, next_tries)
            if i + 1 < rows:  dfs(i+1, j, next_tries)
            if j - 1 >= 0:  dfs(i, j-1, next_tries)
            if j + 1 < cols:  dfs(i, j+1, next_tries)
            boolarray[i][j] = True

        for i in range(rows):
            for j in range(cols):
                dfs(i, j, trie)    
        return res





    # def findWords(self, board, words):
    #     """
    #     :type board: List[List[str]]
    #     :type words: List[str]
    #     :rtype: List[str]
    #     """
    #     res = []
    #     maps = []
    #     def exist( board, word):
    #         if not board: return False
    #         rows = len(board)
    #         cols = len(board[0])
    #         size = len(word)
    #         t =  '\0'
    #         def fcuk(i, x, y):
    #             if i == size:
    #                 return (True,0)
    #             if x>=rows or x<0 or y>=cols or y <0 or word[i]!=board[x][y]:
    #                 return (False,i+1)
    #             t = board[x][y]
    #             board[x][y]='\0'
    #             if fcuk(i+1,x+1,y) or fcuk(i+1,x,y+1) or fcuk(i+1,x-1,y) or fcuk(i+1,x, y-1):
    #                 board[x][y] = t
    #                 return (True,i+1)
    #             else:
    #                 board[x][y] = t
    #             return (False,i)
    #         for pre in maps:
    #             if word.find(pre) != -1:
    #                 return 
                
    #         for i in range(rows):
    #             for j in range(cols):
    #                 if fcuk(0,i,j):
    #                     res.append(word)
    #                     return
    #     for w in words:
    #         if w not in res:
    #             exist(board, w)
    #     return res
        
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """    
        if not wordDict: return []
        if not s: return []
        d = [[False,[]]for i in range(len(s)+1)]
        d[0][0]= True
        for i in range(len(s)):
            if d[i][0]:
                for w in wordDict:
                    if w == s[i:i+len(w)]:
                        d[i+len(w)][0]=True
                        d[i+len(w)][1].append(i)
        # for i in range(len(s)):
        #     for w in wordDict:
        #         ll = i-len(w)+1
        #         if w == s[ll:i+1] and d[ll]:
        #             d[i+1][0] = True
        #             d[i+1][1].append(ll)

        maps={}
        l = len(s)
        def fuck(i):
            if i in maps:
                return maps[i]
            maps[i] = []
            for nexts in d[i][1]:
                part = fuck(nexts)
                strs = s[nexts:i]
                if nexts == 0:
                    maps[i].append(strs)
                else:
                    for x in part:
                        maps[i].append(x + " " + strs)
            return maps[i]
        return fuck(l)


        # if not wordDict: return []
        # if not s: return []
        # d = [[False,[]]for i in range(len(s)+1)]
        # d[0][0]= True
        # for i in range(len(s)):
        #     for w in wordDict:
        #         if w == s[i-len(w)+1:i+1] and d[i-len(w)+1]:
        #             d[i+1][0] = True
        #             d[i-len(w)+1][1].append(i+1)
        
        # maps={}
        # l = len(s)
        # def fuck(i):
        #     if i in maps:
        #         return maps[i]
        #     maps[i]=[]
        #     for nexts in d[i][1]:
        #         part = fuck(nexts)
        #         strs = s[i:nexts]
        #         if nexts!= l:
        #             if part:
        #                 for x in part:
        #                     maps[i].append(strs+" "+ x)
        #         else:
        #             maps[i].append(strs)
        #     return maps[i]
        # return fuck(0)

    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        lhs = newInterval.start
        rhs = newInterval.end
        left = []
        first = True
        second = True
        fck = True
        for i in intervals:
            if i.end < lhs:
                left.append(i)
            elif i.start >rhs:
                if  fck:
                    left.append(Interval(lhs,rhs))
                    second  = False
                    fck =False
                left.append(i)
            else:
                if first:
                    lhs = min(lhs,i.start)
                    first= False
                rhs = max(rhs, i.end)
        if second:
            left.append(Interval(lhs,rhs))
        return left
                
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        h = l1 = ListNode(0)
        l1.next = head
        tmp = None
        while h and h.next:
            if h.next.val>=x: break
            h = h.next
        pre = h
        h = h.next
        while h and h.next:
            if h.next.val < x:
                tmp = h.next
                h.next = h.next.next
                tmp.next = pre.next
                pre.next = tmp
                pre = pre.next
            else:
                h = h.next
        return l1.next



    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix: return False
        if not matrix[0]: return False
        rows = len(matrix)
        beg = 0
        end = rows-1
        mid = int((beg+end)/2)
        while beg != end:
            if beg-end== -1:
                if matrix[end][0] > target:
                    mid = beg
                else:
                    mid = end
                break
            if matrix[mid][0] == target: return True
            elif matrix[mid][0] > target: 
                end = max(mid-1,beg)
            else:
                beg = mid
            mid = int((beg+end)/2)
        realrows = mid
        cols = len(matrix[0])
        beg = 0
        end = cols-1
        mid = int((beg+end)/2)
        while beg!=end:
            if matrix[realrows][mid] == target: return True
            elif matrix[realrows][mid] > target:
                end =  max(mid-1,beg)
            else:
                beg = min(mid +1, end)
            mid = int((beg+end)/2)
        return matrix[realrows][beg] == target

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """

        if not board:
            return False
        rows = len(board)
        cols = len(board[0])
        size = len(word)
        t =  '\0'
        def fcuk(i, x, y):
            if i == size:
                return True
            if x>=rows or x<0 or y>=cols or y <0 or word[i]!=board[x][y]:
                return False
            t = board[x][y]
            board[x][y]='\0'
            if fcuk(i+1,x+1,y) or fcuk(i+1,x,y+1) or fcuk(i+1,x-1,y) or fcuk(i+1,x, y-1):
                return True
            else:
                board[x][y] = t
            return False
        for i in range(rows):
            for j in range(cols):
                if fcuk(0,i,j):
                    return True
        return False
    
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        a=a[::-1]
        b = b[::-1]
        la = len(a)
        lb = len(b)
        ai = 0
        ai = 0
        c =""
        nexts = 0
        while ai < la and ai < lb:
            if a[ai] == '1' and b[ai] == '1': 
                c=(c+'0' if nexts == 0 else c+'1')
                nexts = 1
            elif a[ai] == '0' and b[ai] == '0':
                c=(c+'0' if nexts == 0 else c+'1')
                nexts = 0
            else:
                c=(c+'1' if nexts == 0 else c+'0')
            ai+=1
        while ai < lb:
            if b[ai] == '1' and nexts == 1:
                c+='0'
            elif b[ai] == '0' and nexts == 0:
                c+='0'
            else:
                c+='1'
                nexts = 0
            ai+=1
        while ai < la:
            if a[ai] == '1' and nexts == 1:
                c+='0'
            elif a[ai] == '0' and nexts == 0:
                c+='0'
            else:
                c+='1'
                nexts = 0
            ai+=1
        if nexts == 1:
            c+='1'
        return c[::-1]

    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid: return 0
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])
        first = True
        if obstacleGrid[rows-1][cols-1] == 1:
            return 0
        for j in range(cols-1, -1, -1):
            if first and obstacleGrid[rows-1][j] == 0:
                obstacleGrid[rows-1][j]=1
            else:
                first = False
                obstacleGrid[rows-1][j] = 0
        first= True
        for i in range(rows-2, -1, -1):
            if first and obstacleGrid[i][cols-1] == 0:
                obstacleGrid[i][cols-1] = 1
            else:
                first = False
                obstacleGrid[i][cols-1] = 0
                
        for j in range(cols-2, -1, -1):
            for i in range(rows-2, -1, -1):
                if obstacleGrid[i][j] == 1:
                    obstacleGrid[i][j] = 0
                else:
                    obstacleGrid[i][j]= obstacleGrid[i][j+1] + \
                        obstacleGrid[i+1][j]
        return obstacleGrid[0][0]
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s: return 0
        lens = len(s)
        res = 0
        k= lens-1
        while k>=0:
            if s[k]==' ':
                k-=1
            else:
                break
        while k>=0:
            if s[k]==' ':
                break
            res+=1
            k-=1
        return res
            
            
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        maps = {}
        sums = 0
        for i in nums:
            if i not in maps:
                left = maps.get(i-1,0)
                right = maps.get(i+1,0)
                sum = left + right + 1
                res = max(sum, res)
                maps[i] = sum
                maps[i-left] = sum
                maps[i+right] = sum
        return res
        for i in nums:
            if i not in maps:
                left = maps.get(i-1,0)
                right = maps.get(i+1,0)
                sum = left + right + 1
                res = max(sum, res)
                maps[i] = sum
                maps[i-left] = sum
                maps[i+right] = sum
        return res

    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        lists = [p for p in path.split("/")]
        stack=[]    
        for p in lists:
            if p == "/" or p == "" or p == ".":
                continue
            if p == "..":
                if stack:
                    stack.pop()
            else:
                stack.append(p)
        return "/" + "/".join(stack)


    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """

        lens = len(ratings)
        if lens <= 1:
            return lens
        nums = [1]* lens
        res = 0
        for i in range(1,lens):
            if ratings[i] > ratings[i-1]:
                nums[i] = nums[i-1] + 1
        for i in range(lens-1,0,-1):
            if ratings[i] < ratings[i-1]:
                nums[i-1] = max(nums[i]+1, nums[i-1])
        for number in nums:
            res += number
        return res          

    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        mat = [[0 for x in range(n)] for y in range(n)]
        end = n*n
        k = 1
        i = 0
        while k <= end:
            j = i
            while j < n-i:
                mat[i][j] = k
                j += 1
                k += 1
            j = i+1
            while j < n-i:
                mat[j][n-i-1] = k
                j += 1
                k += 1
            j = n-i-2
            while j > i:
                mat[n-i-1][j] = k
                j -= 1
                k += 1
            j = n - i -1
            while j > i:
                mat[j][i] = k
                j -= 1
                k += 1
            i+=1

        return mat


class MinStack(object):
    
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stk = []
        self.min = 2**31-1

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stk.append(x)
        if x < self.min:
            self.min = x
    def pop(self):
        """
        :rtype: void
        """
        tmp = self.stk.pop()
        if tmp == self.min:
            if self.stk:
                res = self.stk[0]
                for nums in self.stk:
                    if res > nums:
                        res = nums
                self.min = res
            else:
                self.min = 2**31-1
    def top(self):
        """
        :rtype: int
        """
        return self.stk[len(self.stk)-1]
    def getMin(self):
        """
        :rtype: int
        """
        if not self.stk: return 
        return self.min




# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

class WordDictionary(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root={}

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        root = self.root
        for c in word:
            root = root.setdefault(c, {})
        root['#'] = '#'

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        def help(root, i):
            if i == len(word): return '#' in root
            if word[i] != '.':
                if word[i] not in root:
                    return False
                else:
                    root = root[word[i]]
                    return help(root, i+1)
            else:
                for nodes in root.values():
                    if nodes != '#' and help(nodes, i+1):
                        return True
                return False
        return help(self.root, 0)
                
        

class Node:
    def __init__(self, key, val):
        self.val = val
        self.key = key
        self.prev = None
        self.next = None
    def insertBefore(self, node):
        self.prev = node.prev
        node.prev.next = self
        self.next = node
        node.prev = self
    def delself(self):
        self.prev.next = self.next
        self.next.prev = self.prev
class LRUCache:

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.maps = {}
        self.capacity = capacity
        self.beg = Node(0, 0)
        self.tail = Node(0, 0)
        self.beg.next = self.tail
        self.tail.prev = self.beg

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.maps:
            node = self.maps[key]
            node.delself()
            node.insertBefore(self.tail)
            # node.prev.next = node.next
            # node.next.prev = node.prev

            # node.prev = self.tail.prev
            # self.tail.prev.next = node
            # node.next = self.tail
            # self.tail.prev = node
            return node.val
        else:
            return -1
    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        node = Node(key, value)
        self.maps[key] = node

        if len(self.maps) > self.capacity:
            deleted = self.beg.next
            self.beg.next = self.beg.next.next
            self.beg.next.prev = self.beg
            del self.maps[deleted.key]



# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
test = Solution()
# rr = Interval(2,5)
# l1 = Interval(1,3)
# l2 = Interval(6,9)
# l=[]
# l.append(l1)
# l.append(l2)
# print(test.findLadders("red","tax",["ted","tex","red","tax","tad","den","rex","pee"]))
# print(test.findLadders("hot","dog",["hot","dog"]))
# print(test.findLadders("hit", "cog", ["hot","dot","dog","lot","log","cog"]))
# print(test.isScramble('abc','bca'))
# print(test.removeDuplicateLetters("cbacdcbc"))
# print(test.left([2,3,1,4,4,2]))
# print(test.findSubstring("wordgoodgoodgoodbestword",["word","good","best","good"]))
# print(test.minWindow("ab","b"))
# a = ["..9748...","7........",".2.1.9...","..7...24.",".64.1.59.",".98...3..","...8.3.2.","........6","...2759.."]
# (test.solveSudoku(a))
# print(a)
# print(test.canFinish(2,[[1,0]]))
# print(test.isValid('()()'))
# print(test.isValid("()(()"))
# print(test.lengthOfLIS([10,9,2,5,3,7,101,18]))
# print(test.shortestPalindrome('aacecaaa'))
# print(test.shortestPalindrome('abcd'))
# print(test.shortestPalindrome('aac'))
# print(test.shortestPalindrome('ac'))
# print(test.shortestPalindrome('abbacd'))
# print(test.palindromePairs(["abcd","dcba","lls","s","sssll"]))
# abcd
# ""
def quanpaixu(a):
    if len(a) == 0:
        return [[]]
    res = []
    for i in range(len(a)):
        rest = quanpaixu(a[:i]+a[i+1:])
        res+=([[a[i]] + r for r in rest])
    return res
    # return [ [ a[i]+ ] ]
print(quanpaixu([1,2,3,4]))
# def qsort(a, i, j):
#     if j <= i+1: return 
#     midValue = a[i]
#     beg = i+1
#     end = j
#     while beg < end:
#         if a[beg] < midValue:
#             beg+=1
#         else:
#             end-=1
#             a[beg], a[end] = a[end], a[beg]
#     # if a[beg] > a[end]
#     a[i], a[beg-1] = a[beg-1], a[i]
#     qsort(a, i, beg-1)
#     qsort(a, beg, j)
# import random
# test1 = [random.randint(1, 1000) for _ in range(100)]
# sortednum = test1
# print(test1)
# qsort(test1 ,0, len(test1))
# print(test1)
# if test1 == sortednum:
#     print('ojbk')
        
# head = stringToListNode('[1,2,1]');
# ret = Solution().isPalindrome1(head)
# print(ret)
# head = stringToListNode('[1,2,2,1]');
# ret = Solution().isPalindrome1(head)
# print(ret)
# head = stringToListNode('[1,2,3,1]');
# ret = Solution().isPalindrome1(head)
# print(ret)
# head = stringToListNode('[1,2,3]');
# ret = Solution().isPalindrome1(head)
# print(ret)
# head = stringToListNode('[1]');
# ret = Solution().isPalindrome1(head)
# print(ret)
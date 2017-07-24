from collections import deque
# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e



class Solution(object):
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
        for i in range(heights):
            tmp = heights[i]*(left[i] + right[i] + 1)
            res = max(res, tmp)
        return res

    def left(self, heights):
        distance = [0]*len(heights)
        for i in range(len(heights)):
            j = i -1
            t = 0
            while j >= 0:
                if heights[i] <= heights[j]:
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
            while j < len(heights):
                if heights[i]<=heights[j]:
                    t = t + distance[j] + 1
                    j = j + distance[j] + 1
                else: 
                    break
                j+=1                
            # for j in range(i+1,len(heights)):
            #     if heights[i]<=heights[j]:
            #         t+=1
            #     else:
            #         break
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
        def construct_dict(word_list):
            def compare(lhs, rhs):
                k = 0
                for i in range(len(lhs)):
                    if lhs[i] != rhs[i]:
                        k+=1
                    if k > 1:
                        return -1
                return k
            d = {}
            
            for lhs in word_list:
                for rhs in word_list:
                    if compare(lhs,rhs) == 1:
                        tmp = d.setdefault(lhs, set())
                        tmp.add(rhs)
                        tmp = d.setdefault(rhs, set())
                        tmp.add(lhs)
                        
                # for i in range(len(word)):
                #     s = word[:i]+"_"+word[i+1:]
                #     d[s] = d.get(s, []) + [word]
            return d
        lens = 0
        res = []
        if endWord not in wordList: return []
        if beginWord not in wordList: wordList.append(beginWord)

        dict_words = construct_dict(wordList)

        queue, visted = deque([(beginWord, 1,[beginWord])]), set()
        maps ={1:set([beginWord])}

        l = 0
        while queue:
            # print(queue)
            word, steps, paths = queue.popleft()
            if word == endWord:
                res.append(paths)
                lens = steps
                break
            if l != lens:
                for vals in maps:
                    visted.add(vals)
                l = lens
            if word not in visted:
                next_words = dict_words.get(word, [])
                for nexts in next_words:
                    if nexts != word and nexts not in visted:
                        queue.append((nexts, steps+1, paths+[nexts]))
                        tmp = maps.setdefault(steps,set())
                        tmp.add(nexts)
        while queue:
            word, steps, paths = queue.popleft()
            if steps != lens:
                break
            if word == endWord:
                res.append(paths)
        return res

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
# print(test.isScramble('abc','bca'))
# print(test.removeDuplicateLetters("cbacdcbc"))
print(test.left([2,3,1,4,4,2]))
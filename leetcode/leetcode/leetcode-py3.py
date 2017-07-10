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
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        res = []
        maps = []
        def exist( board, word):
            if not board: return False
            rows = len(board)
            cols = len(board[0])
            size = len(word)
            t =  '\0'
            def fcuk(i, x, y):
                if i == size:
                    return (True,0)
                if x>=rows or x<0 or y>=cols or y <0 or word[i]!=board[x][y]:
                    return (False,i+1)
                t = board[x][y]
                board[x][y]='\0'
                if fcuk(i+1,x+1,y) or fcuk(i+1,x,y+1) or fcuk(i+1,x-1,y) or fcuk(i+1,x, y-1):
                    board[x][y] = t
                    return (True,i+1)
                else:
                    board[x][y] = t
                return (False,i)
            for pre in maps:
                if word.find(pre) != -1:
                    return 
                
            for i in range(rows):
                for j in range(cols):
                    if fcuk(0,i,j):
                        res.append(word)
                        return
        for w in words:
            if w not in res:
                exist(board, w)
        return res
        
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

test = Solution()
# rr = Interval(2,5)
# l1 = Interval(1,3)
# l2 = Interval(6,9)
# l=[]
# l.append(l1)
# l.append(l2)
print(test.wordBreak("catsanddog",["cat","cats","and","sand","dog"]))
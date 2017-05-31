# from collections import Counter
import collections

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


# import random
# from math import log
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import random
import Queue


class Solution(object):
    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def fuck():
            stk = []
            stk.append([[],[],[]])
            while stk:
                a = stk.pop()
                queen=a[0]
                xy_dif=a[1]
                xy_sum=a[2]
                length = len(queen)
                if length == n:
                    result.append(queen)
                for j in range(n):
                    if j not in queen and length-j not in xy_dif and length+j not in xy_sum:
                        stk.append([queen+[j], xy_dif + [length-j] , xy_sum + [length+j]])
        result = []
        fuck()
        return [["."*i + "Q" + "."*(n-i-1) for i in sol]for sol in result]
                

        # def DFS(queen, xy_dif, xy_sum):
        #     length = len(queen)
        #     if length == n:
        #         result.append(queen)
        #         # result.append(queen[:])
        #         return
        #     for j in range(n):
        #         if j not in queen and length-j not in xy_dif and length+j not in xy_sum:
        #             DFS(queen+[j], xy_dif + [length-j] , xy_sum + [length+j])
        #             # queen +=[j]
        #             # xy_dif +=[length-j]
        #             # xy_sum +=[length+j]
        #             # DFS(queen,xy_dif,xy_sum)
        #             # queen.remove(j)
        #             # xy_dif.remove(length-j)
        #             # xy_sum.remove(length+j)
        # result = []
        # DFS([], [], [])
        # return [["."*i + "Q" + "."*(n-i-1) for i in sol]for sol in result]


        
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        g=[1 for i in range(n)]
        for i in range(n):
            for j in range(i):
                pass


    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        row = len(matrix)
        if row == 0:
            return
        col = len(matrix[0])
        if col == 0:
            return
        
        rowline = [0 for x in range(row)]
        colline = [0 for x in range(col)]
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == 0:
                    rowline[i] = 1
                    colline[j] = 1

                
        for i in range(row):
            if rowline[i] == 1:
                matrix[i] = [0 for x in range(col)]
            for j in range(col):
                if colline[j] == 1:
                    matrix[i][j] = 0



    # def subsets(self, nums):
    #     """
    #     :type nums: List[int]
    #     :rtype: List[List[int]]
    #     """
    #     res =[[]]
    #     for num in sorted(nums):
    #         res+=[[num] + x for x in res]
    #     return res
    # def subsets(self, nums):
    #     res = []
    #     nums.sort()
    #     for i in xrange(1<<len(nums)):
    #         tmp = []
    #         for j in xrange(len(nums)):
    #             if i & 1 << j:  # if i >> j & 1:
    #                 tmp.append(nums[j])
    #         res.append(tmp)
    #     return res
    def subsets(self, nums):
        res = []
        self.dfs(sorted(nums), 0, [], res)
        return res
    def dfs(self, nums, index, path, res):
        res.append(path)
        for i in xrange(index, len(nums)):
            self.dfs(nums, i+1, path+[nums[i]], res)



    maps={}
    def helper(self, m, n):
        if (m,n) in self.maps:
            return self.maps[(m,n)]
        else:
            if m ==1:
                return 1
            if n==1:
                return 1
            self.maps[(m,n)]=self.helper(m-1,n)+self.helper(m,n-1)
            return self.maps[(m,n)]
    def uniquePaths(self, m, n):
        self.maps[(1,1)]=1
        res = self.helper(m,n)
        return res
    # def helper(self, m, n,maps):
    #     if (m,n) in maps:
    #         return maps[(m,n)]
    #     else:
    #         if m ==1:
    #             return 1
    #         if n==1:
    #             return 1
    #         maps[(m,n)]=self.helper(m-1,n,maps)+self.helper(m,n-1,maps)
    #         return maps[(m,n)]
    # def uniquePaths(self, m, n):
    #     """
    #     :type m: int
    #     :type n: int
    #     :rtype: int
    #     """
    #     maps={}
    #     maps[(1,1)]=1
    #     res = self.helper(m,n,maps)
    #     return res


    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        if n == 0:
            return [0]
        if n == 1:
            return [0,1]
        res =[0 for x in range(2**n)]
        res[1]=1
        for i in range(1,n):
            res[2**(i):2**(i+1)]=[x + x**i for x in res[:2**(i)][::-1]]
        return res

    def isInterleave(self, s1, s2, s3):
        """
        :type s1: str
        :type s2: str
        :type s3: str
        :rtype: bool
        """
        ls1 = len(s1)
        ls2 = len(s2)
        ls3 = len(s3)
        def fuck(s1,l1,s2,l2,s3,l3):
            if  l1 == ls1 : return s2[l2:]==s3[l3:]
            if l2 == ls2: return s1[l1:]==s3[l3:]
            if l3 == ls3: return False

            if s1[l1]!=s2[l2]:
                if s3[l3]==s1[l1]:
                    return fuck(s1,l1+1,s2,l2,s3,l3+1)
                if s3[l3] == s2[l2]:
                    return fuck(s1,l1,s2,l2+1,s3,l3+1)
                else:
                    return False
            else:
                if s3[l3]==s1[l1]:
                    return fuck(s1,l1,s2,l2+1,s3,l3+1) or  fuck(s1,l1+1,s2,l2,s3,l3+1)
                else:
                    return False
            
        return fuck(s1,0,s2,0,s3,0)



    def recoverTree(self, root):
        stk = []
        p = root
        node1 = None
        node2 = None 
        prev = TreeNode(-2**32)
        notfirst = False
        while stk or p:
            while p:
                stk.append(p)
                p = p.left
            if stk:
                node = stk.pop()
                if not node1:
                    if prev.val>=node.val:
                        node1 = prev
                if node1:
                    if prev.val>=node.val:
                        node2 = node 
                prev = node
                p = node.right
        node1.val, node2.val = node2.val, node1.val
        
# class BSTIterator(object):
#     def __init__(self, root):
#         self.stk = []
#         p = root
#         while p:
#             self.stk.append(p)
#             p = p.left

#     def hasNext(self):
#         if self.stk:
#             return True

#     def next(self):
#         node = self.stk.pop()
#         p = node.right
#         while p:
#             self.stk.append(p)
#             p = p.left
#         return node.val
##################################
    def kthSmallest(self, root, k):
        i = k
        stk = []
        p = root

        while stk or p:
            while p:
                stk.append(p)
                p = p.left
            if stk:
                node = stk.pop()
                if i==1:
                    return node.val
                else:
                    i -=1
                p = node.right
        return -1
    def postorderTraversal(self, root):
        if not root:
            return []
        stk = [root]
        res = []
        p = root
        while stk:
            node = stk.pop()
            res.append(node.val)
            if node.left: stk.append(node.left)
            if node.right: stk.append(node.right)
        return res[::-1]          
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        stk = []
        p = root
        res = []
        while stk or p:
            while p:
                stk.append(p)
                p = p.left
            if stk:
                node = stk.pop()
                res.append(node.val)
                p = node.right
        if res == sorted(res):
            return True
        return False

    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        stk = [root]
        res = []
        p = root
        while stk:
            node = stk.pop()
            res.append(node.val)
            if node.right: stk.append(node.right)
            if node.left: stk.append(node.left)
        return res    
        # stk = []
        # res = []
        # p = root
        # while stk or p:
        #     while p:
        #         res.append(p.val)
       #          stk.append(p.right)
        #         p = p.left
        #         
        #     if stk:
        #         p = stk.pop()
        # return res


    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stk = []
        p = root
        res = []
        while stk or p:
            while p:
                stk.append(p)
                p = p.left
            if stk:
                node = stk.pop()
                res.append(node.val)
                p = node.right
        return res

        
        # s = []
        # def inorder(root):
        #     if root.left: inorder(root.left)        
        #     s.append(root.val)
        #     if root.right: inorder(root.right)
        # if root:
        #     inorder(root)
        # return s
    def magicalString(self, n):
        """
        :type n: int
        :rtype: int
        """
        a = [1,2,2]
        end = 2
        while len(a) < n:
            a += [(3-a[-1])] * a[end]
            end += 1
        return a[:n].count(1)
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        return sorted(nums)[len(nums)//2]
    def findDiagonalOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """


    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
    def nextGreaterElements(self, nums):
        stk = []
        res =[-1]*len(nums)
        for i in range(len(nums)) * 2:
            while stk and nums[stk[-1]] < nums[i]:
                res[stk.pop()] = nums[i]
            stk.append(i)
        return res

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        c = collections.Counter(nums)
        return heapq.nlargetest(k, c, c.get)
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        l = []
        def inorder(node):
            if node.left: bfs(node.left)
            l.append(node.val)
            if node.right: bfs(node.right)
        inorder(root)
        return min([abs(l[i] - l[i+1]) for i in range(len(l)-1)])        


    def get(self,root,maps):
            if root not in maps:
                maps[root] = root.val +self.get(root.left,maps)+self.get(root.right, maps)
            return maps[root]
    def fuck(self, root,maps):
        if not root:
            return 0
        return abs(maps[root.left]- maps[root.right])+self.fuck(root.left,maps)+self.fuck(root.right,maps)
    def findTilt(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        maps={None:0}
        self.get(root,maps)
        return self.fuck(root,maps)


    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        if num == 0: return "0"
        elif num < 0: num += 2**32
        converthex, res =["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"], ""
        while num:
            res = converthex[num%16] + res
            num = num//16
        return res
        


    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head  or not head.next:
            return False
        slow = head.next
        fast = head.next.next
        while slow != fast  :
            if not fast or not fast.next:
                return False
            else:
                fast = fast.next.next
                slow = slow.next 
        return True
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        tmp = self.generate(numRows-1)
        tmp1=[1]
        for i in range(numRows - 2):
            tmp1.append(tmp[numRows-2][i]+tmp[numRows-2][i+1])
        tmp1.append(1)
        tmp.append(tmp1)
        return tmp
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        if not root.left and not root.right:
            return [str(root.val)]
        treepaths = [str(root.val) + '->' + path for path in self.binaryTreePaths(root.left)]
        treepaths += [str(root.val) + '->' + path for path in self.binaryTreePaths(root.right)]
        return treepaths
        
    def helper(self, root, sum, store, pre):
        if not root:    
            return 0
        pre = pre + root.val

        res = int(pre == sum) +(store[pre-sum] if (pre - sum) in store else 0)
        if pre in store:
            store[pre]+=1
        else:
            store[pre] = 1

        res += self.helper(root.left, sum, store, pre) + self.helper(root.right, sum, store, pre)
        store[pre]-=1
        return res 


    def pathSum(self, root, sum):
        store = {}
        return self.helper(root, sum, store, 0)
    # def find_path(self, root, target):
    #     if root:
    #         return int(root.val == target) + self.find_path(root.left, target-root.val) 
    #             + self.find_path(root.right, target-root.val)
    #     return 0
    # def pathSum(self, root, sum):
    #     """
    #     :type root: TreeNode
    #     :type sum: int
    #     :rtype: int
    #     """
    #     if root:
    #         return self.find_path(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)
    #     return 0

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <=0:
            return 0
        if n == 1:
            return 1
        if n == 2:
            return 2
        lhs = 1
        rhs = 2
        allnum = 0
        for i in range(3, n+1):
            allnum = lhs + rhs
            lhs = rhs
            rhs = allnum
        return allnum
                    
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        diff = reduce(lambda a, b: a^b,nums,0)
        diff &= -diff #the last  1 
        res =[0, 0]
        for i in nums:
            if i & diff == 0:
                res[0] ^= i;
            else:
                res[1] ^= i;
        return res

        
    def frequencySort(self, s):
        c = collections.Counter(s)
        lists = sorted((c[i],i) for i in c)
        # print lists
        return reduce(lambda a, b:b[1]*b[0]+a ,lists,"")

        # c = collections.Counter(s)
        # return reduce(lambda a,b: b[1]*b[0]+a, sorted((c[i],i) for i in c), '')
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 2:
            return 1
        if n == 3:
            return 2
        if n == 4:
            return 4
        res = 1
        while n > 4:
            res *= 3
            n -= 3
        # n 4 3 2
        return res * n 
    
    # def __init__(self, nums):
    #     """
    #     :type nums: List[int]
    #     """
    #     self.nums = nums

    def reset(self):
        """
        Resets the array to its original configuration and return it.
        :rtype: List[int]
        """
        return self.nums        

    def shuffle(self):
        """
        Returns a random shuffling of the array.
        :rtype: List[int]
        """
        nums = self.nums[:]
        for i in range(len(nums)):
            j = random.randrange(i, len(nums))
            nums[i], num[j] = nums[j],nums[i]
        return nums


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        cur = None
        while head:
            cur = head
            head = head.next
            cur.next = prev
            prev = cur
        return cur
            
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        #(0,1)  0 not include root , 1 may include root
        def dfsrob(self, node):
            if node == None:
                return (0, 0)
            l = self.dfsrob(node.left)
            r = self.dfsrob(node.right)
            return (l[1]+r[1], max(l[1]+r[1], l[0]+r[0]+ node.val))
        return self.dfsrob(root)[1]

        
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        res =[]
        i = len(num1)-1
        j = len(num2)-1
        tmp = 0
        inc =0
        while i!=-1 and j!=-1:
            tmp = int(num1[i]) +int(num2[j])+inc
            if tmp>=10:
                tmp -= 10
                inc = 1
            else:
                inc = 0
            res.append(tmp)
            i-=1
            j-=1
        while i!=-1:
            tmp = int(num1[i]) + inc 
            if tmp>=10:
                tmp -= 10
                inc = 1
            else:
                inc = 0
            res.append(tmp)
            i-=1
        while j!=-1:
            tmp = int(num2[j]) + inc
            if tmp>=10:
                tmp -= 10
                inc = 1
            else:
                inc = 0
            res.append(tmp)
            j-=1
        if inc ==1:
            res.append(1)
            
        ans = [str(x) for x in reversed(res)]
        ans="".join(ans)
        return ans


    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # pre, cur, tmp= 0, 0, 0
        # for i in range(1,len(nums)):
        #     if i%2 == 0:
        #         pre = max(pre+nums[i], cur)
        #     else:
        #         cur = max(cur+nums[i], pre)
            
        # a = max(pre, cur)
        # pre, cur, tmp= 0, 0, 0
        # for i in range(0,len(nums)-1):
        #     if i%2 == 0:
        #         pre = max(pre+nums[i], cur)
        #     else:
        #         cur = max(cur+nums[i], pre)
        # b = max(pre, cur)
        # return max(a,b)
        
        # pre, cur, tmp= 0, 0, 0
        # for i in nums:
        #     tmp = max(pre+i,cur)
        #     pre = cur
        #     cur = tmp
        # return tmp

        # if len(nums)==0:
        #     return 0
        # elif len(nums)==1:
        #     return nums[1]
        # elif len(nums)==2:
        #     return max(nums[1],nums[0])
        # else:
        #     ans1 = nums[0] + self.rob(nums[2:])
        #     ans2 = nums[1] + self.rob(nums[3:])
        #     return max(ans1, ans2)


    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        maxres = 0
        for num in nums:
            if num>=0:
                res+=num
                if res > maxres:
                    maxres = res
            elif res+num>0:
                res =res+num
            else:
                res = 0
        
        return maxres if maxres!=0 else max(nums)
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return (n>0) and (n & (n-1))==0
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n <= 0 :
            return False
        else:
            root = log(n,3)
            root_round = round(root)
            diff = abs(root - root_round)
            return True if diff <= 1e-10 else False

    def sumofdigits(self, n):
        res = 0
        while n:
            res+=(n%10)**2
            n=n/10
        return res
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        
        
        slow = self.sumofdigits(n)
        fast = slow
        fast = self.sumofdigits(fast)
        while fast!=slow:
            slow = self.sumofdigits(slow)
            fast = self.sumofdigits(fast)
            fast = self.sumofdigits(fast)
        if slow ==1:
            return True
        else:
            return False


    # def plusOne(self, digits):
    #     """
    #     :type digits: List[int]
    #     :rtype: List[int]
    #     """
    #     num = reduce(lambda x, y: x * 10 + y, digits)+1
    #     return [int(i) for i in str(num)]
    # def __init__(self, head):
    #     """
    #     @param head The linked list's head.
    #     Note that the head is guaranteed to be not null, so it contains at least one node.
    #     :type head: ListNode
    #     """
    #     self.node = head
    def getRandom(self):
        """
        Returns a random node's value.
        :rtype: int
        """
        res = self.node.val
        n = 0
        cur = self.node.next
        while cur:
            n +=1
            if random.randint(0, n) == 0:
                res =cur.val
            cur = cur.next
        return res
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(head)
# param_1 = obj.getRandom()
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        maps ={}
        for chars in s:
            maps[chars] = maps.get(chars,0)+1
        res = 0
        one =0
        for v in maps:
            if maps[v]%2==0:
                res +=maps[v]
            else:
                res += maps[v]-1
                one =1
        return res + one 


    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        res = 0
        for p in points:
            camp = {}
            for q in points:
                z=(p[0]- q[0])**2+(p[1]- q[1])**2
                camp[z] = 1 + camp.get(z, 0)
            for k in camp:
                res += camp[k] * (camp[k] - 1)
        return res
                
        
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        return s[:k][::-1] + s[k:2*k]+self.reverseStr(s[2*k:],k) if s else ""

    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        return ['%d:%02d' % (h, m)
                for h in range(12) for m in range(60)
                if (bin(h)).count('1') +  bin(m).count('1') == num]




    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # res1 = ''.join(sorted(s))
        # res2 =''.join(sorted(t))
        # if res1 == res2:
        #     return True
        # else:
        #     return False

        a={}

        for letter in s:
            if letter in a:
                a[letter] += 1
            else:
                a[letter] = 1
        for letter in t:
            if letter in a and a[letter]>0:
                a[letter] -=1
            else:
                return False
        for key,v in enumerate(a):
            if a[v]!=0:
                return False
        return True
        
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        ind = sorted(range(len(nums)), key = lambda x: nums[x])
        for i in range(len(nums)-1):
            j = i + 1
            while j < len(nums) and nums[ind[j]] - nums[ind[i]] <= t:
                if abs(ind[i]-ind[j]) <= k:
                    return True
                j += 1
        return False
    # def containsNearbyAlmostDuplicate(self, nums, k, t):
    #     """
    #     :type nums: List[int]
    #     :type k: int between i nums j
    #     :type t: int between nums[i] and nums[j]
    #     :rtype: bool
    #     """
    #     bucket={}
    #     for i,v in enumerate(nums):
    #         bucketnums = v/t if t else v
    #         offset = 1 if  t else 0
    #         for buckidx in range(bucketnums - offset, bucketnums + offset + 1):
    #             if buckidx in bucket and abs(v - bucket[buckidx]) <= t:
    #                 return True
    #         bucket[bucketnums] = v
    #         if len(bucket) > k:
    #             del bucket[nums[i - k]/t if t else nums[i-k]]
    #     return False


    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        res ={}
        for i in xrange(len(nums)):
            x = nums[i]
            if x in res:
                last = res[x]
                if i -last <= k:
                    return True
                else:
                    res[x] = i
            else:
                res[x] = i
        return False

    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        res={}
        for x in nums:
            if x in res:
                return True
            else:
                res[x] =1
        return False
    def convertToBase7(self, num):
        """
        :type num: int
        :rtype: str
        """
        lists=""
        pos =True
        if num == 0:
            return "0"
        if num <0:
            pos =False
            num = -num
        while num >0:
            a=num%7
            num/=7
            lists=str(a)+lists
        if not pos:
            lists='-'+lists
        return lists

    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans =[[]]
        for n in nums:
            ans =[ l[:i] + [n]+l[i:]
                for l in ans
                for i in xrange((l+[n]).index(n) + 1 )]
        return ans
        # return reduce(lambda a,n:[l[:i]+[n]+l[i:]for l in a for i in xrange((l+[n]).index(n)+1)],nums,[[]])


    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        left,right,water,minheight = 0,n-1,0,0
        while left < right:
            while left <right and height[left]<= minheight:
                water += minheight - height[left]
                left +=1
            while right >left and height[right] <= minheight:
                water += minheight - height[right]
                right -=1
            minheight = min(height[left], height[right])
        return water
    
    def kmpPre(self, needle):
            nextlist = [0 for x in len(needle)]
            k = 0
            for i in range(1,len(needle)):
                while k > 0 and needle[i] != needle[k]:
                    k = nextlist[k-1]
                if needle[i] == needle[k]:
                    k=k+1
                nextlist[i]=k
            return nextlist
        
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle) == 0:
            
            return 0
        
        nextlist = self.kmpPre(needle)

        k = 0
        for i in range(0, len(haystack)):
            while k > 0 and haystack[i] != needle[k]:
                k = nextlist[k-1]

            if haystack[i] == needle[k]:
                k=k+1
            if k == len(needle) :
                return i-k+1
        return -1



    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums)==0:
            return -1
        low, high = 0,len(nums)-1
        mid = (low+high)/2
        
        while low <mid:
            if nums[low] < nums[mid] and nums[low]<=target and target <= nums[mid]:
                high = mid
            elif nums[low]> nums[mid] and (nums[low]<=target or target<=nums[mid]):
                high = mid
            else:
                low = mid+1
            mid = (low+high)/2
                
        if nums[low]==target:
            return low
        elif low+1<len(nums)and nums[low+1] == target:
            return low+1
        else:
            return -1
        
            

    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic ={}
        for str in strs:
            s = ''.join(sorted(str))

            if s in dic:
                dic.get(s).append(str)
            else:
                dic[s] = [str]
        return dic.values()
             
            
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        else:
            res  = len(nums)
            for i in range(len(nums)):
                if nums[i]>=target:
                    res = i
                    break
            return res 

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums)<=1:
            return len(nums)
        
        idx = 1

        for i in range(1,len(nums)):
            if (nums[i]!=nums[i-1]):
                nums[idx] = nums[i]
                idx+=1
        return idx
                
                


    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n =len(matrix)
        if n == 0:
            return
        
        for i in range((n)/2):
            for j in range((n+1)/2):
                # tmp = matrix[i][j]
                # matrix[i][j] = matrix[n-j-1][i]
                # matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
                # matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
                # matrix[j][n-i-1]=tmp
                matrix[i][j], matrix[n-j-1][i] , matrix[n-i-1][n-j-1],matrix[j][n-i-1]=matrix[n-j-1][i] , matrix[n-i-1][n-j-1],matrix[j][n-i-1],matrix[i][j]


    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 and l2 and l1.val>l2.val:
            l1,l2 = l2,l1
        if l1:
            l1.next = self.mergeTwoLists(l1.next,l2)
        return l1 or l2

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if digits == "":
            return []
        kvmaps = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }   
        # return reduce(lambda acc ,digit: [x + y 
        # for x in acc for y in kvmaps[digits]], digits, [""])
        return [x + y for x in kvmaps[digits[0]] 
                for y in self.letterCombinations(digits[1:])] 



    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        k=0
        for i in reversed(range(0, len(nums))):
            if i >  0 and nums[i] > nums[i-1] :
                for j in reversed(range(i, len(nums))):
                    if nums[i-1] < nums[j]:
                        nums[i-1], nums[j] = nums[j], nums[i-1]
                        break
                k=i
                break
        nums[k:] = nums[k:][::-1]
                    
                        


    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # return nums and [p[:i] + [nums[0]] +p[i:]
        #     for p in self.permute(nums[1:])
        #     for i in range(len(nums))] or [[]]
        return [[n] + p
                for i ,n in enumerate(nums)
                for p in self.permute(nums[:i]+nums[i+1:])] or [[]]

    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        
        if n == 0:
            return 1
        elif n==1:
            return x
        elif n<0:
            return self.myPow(1/x,-n)
            
        if n % 2 == 1:
            return x * self.myPow(x,n-1)
        else:
            res = self.myPow(x, n/2)
            return res * res


# res = a.permute(['a','b','c','d'])
# for i,v in enumerate(res):
#     print ''.join(v)
# print a.containsDuplicate([1,1])
# print a.containsDuplicate([1,2,1])
# print a.containsDuplicate([])
# print a.containsDuplicate([1,3,4])
# print a.convertToBase7(0)
# print a.convertToBase7(7)
# print a.convertToBase7(3)
# print a.convertToBase7(11)
# print a.convertToBase7(-1)
# print a.convertToBase7(-7)
# print a.convertToBase7(-15)

# b=[[1,2,3],[4,5,6],[7,8,9]]
# a.rotate(b)
# print b
# # print a.myPow(8.88023, 3)
# print " das "
# b=[1,2,2,2,3,3]
# print a.removeDuplicates(b)
# print b
# print a.strStr("w","w")
# print a.strStr("mississippi","issip")
# print a.containsNearbyAlmostDuplicate([1,4,3,2],0,0)

# a = Solution()
# print a.addStrings("1110","99")
# ss="abbbbbbac"
# print a.frequencySort(ss)
# c = collections.Counter(ss)
# print c
# print sorted((c[i],i) for i in c)
# for
# f =["str"+ l for l in ["a",'b','v']]
# print f+["dfdf"]

# a =1 
# b=10
# a,b=a-1,a+1
# print a,b
# nums =[1,2,3,1]
# counts = {}
# for x in nums:
#     counts[x] = counts.setdefault(x, 0) + 1
# print counts
a = Solution()
print a.solveNQueens(4)
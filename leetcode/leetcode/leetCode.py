# from collections import Counter
import collections

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


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
class Solution(object):
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

a = Solution()
# print a.addStrings("1110","99")
ss="abbbbbbac"
print a.frequencySort(ss)
# c = collections.Counter(ss)
# print c
# print sorted((c[i],i) for i in c)
# for
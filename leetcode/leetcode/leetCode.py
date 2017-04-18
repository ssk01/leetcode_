# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
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


a = Solution()
print a.permuteUnique([1,1,2])
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
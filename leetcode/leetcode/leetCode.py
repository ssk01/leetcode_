# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None



class Solution(object):
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
                matrix[i][j], matrix[n-j-1][i] , matrix[n-i-1][n-j-1],matrix[j][n-i-1]=
                 matrix[n-j-1][i] , matrix[n-i-1][n-j-1],matrix[j][n-i-1],matrix[i][j]



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
b=[[1,2,3],[4,5,6],[7,8,9]]
a.rotate(b)
print b
# print a.myPow(8.88023, 3)
print " das "
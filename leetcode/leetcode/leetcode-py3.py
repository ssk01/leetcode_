class Solution(object):
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
print(test.candy([2,1]))
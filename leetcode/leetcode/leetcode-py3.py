class Solution(object):
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
print(test.uniquePathsWithObstacles([[0],[1]]))
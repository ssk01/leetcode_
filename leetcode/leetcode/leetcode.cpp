// #463
class Solution {
public:
     int islandPerimeter(vector<vector<int>>& grid) {
       int result = 0;
       int repeat = 0;
       auto size1 = grid.size();
       for (int i=0; i <size1; i++)
       {
           auto size = grid[i].size();
           for (int j = 0;j < size; j++)
           {
               if (grid[i][j] == 1)
               {
                   result++;
                   if (i!=0&&grid[i-1][j]==1)
                       repeat++;
                   if (j!=0&&grid[i][j-1]==1)
                       repeat++;
               }
           }
       }
       return 4*result -2*repeat;
    }


};


//#292
class Solution {
public:
    bool canWinNim(int n) {
        if (n%4 == 0)
            return false;
        return true;
    }
};

//#485
class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int res=0;
        int max=0;
       
        auto size = nums.size();
        for (int i =0; i<size;++i)
        {
            if ( nums[i])
            {
                res++;
            }
            else 
            {
                if(res>max)
                    max=res;
                res = 0;
            }
        }
        
        return res>max? res:max;
    }
};

//#136
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        auto size = nums.size();
        int result=0;
        for (auto i =0; i< size; i++)
        {
            result ^= nums[i];
        }
        return result;
    }
};

//#448
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int len = nums.size();
        vector<int> result;
        int m; 
        for (auto i =0; i <len;++i)
        {
            m = abs(nums[i]) -1;
            nums[m] = -abs(nums[m]);
        }
        for (auto i = 0; i<len; ++i)
        {
            if (nums[i]>0)
            {
                result.push_back(i+1);
            }
        }
        return result;
        
    }
};

//520
bool detectCapitalUse(string word) {
	bool first = true;
	bool large = true;
	bool fuck = true;
	bool second = true;
	auto size = word.size();
	for (int i = 0; i< size; ++i)
	{

		if (first)
		{
			if (word[i] >= 'a')
			{
				large = false;
			}
			else
			{
				
			}
			first = false;
		}
		else
		{
			if (!large)
			{
				if (word[i]<='Z')
					return false;
			}
			else
			{
				if (word[i]>'Z')
				{
					if (second)
					{
						fuck = true;
						second = false;
					}
					else
					{
						if (!fuck)
						{
							return false;
						}
					}
				}
				else if (word[i] <= 'Z')
				{
					if (second)
					{
						fuck = false;
						second = false;
					}
					if (fuck)
					{
						return false;

					}
				}
			}
		}

	}
	return true;
}

//104
class Solution {
public:
    int maxDepth(TreeNode* root) {
        
        return (root==NULL) ? 0 :max(maxDepth(root->left),maxDepth(root->right))+1;
    }
};
class Solution {
public:
    int maxDepth(TreeNode* root) {
      if (root == NULL)
      {
          return 0;
      }
      queue<TreeNode*> que;
      int res = 0;
      que.push(root);
      while(!que.empty())
      {
          res++;
          for(int i =0, n =que.size(); i<n; i++)
          {
              auto t = que.front();
              que.pop();
              if(t->left!=NULL)
              {
                  que.push(t->left);
              }
              if(t->right!=NULL)
              {
                  que.push(t->right);
              }
          }
      }
        return res;
    }
};

//389
class Solution {
public:
    char findTheDifference(string s, string t) {
        int size =s.size();
        int a =0; int b=0;
        for(int i =0; i<size;i++)
        {
            a += (int)s[i];
            b+=(int)t[i];
        }
        b+=(int)t.back();
        return (char)(b-a);
    }
};

//226
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root!=NULL)
        {
            TreeNode * tmp = root->left;
            root->left = root->right;
            root->right = tmp;
            if (root->left!=NULL)
                invertTree(root->left);
            if (root->right!=NULL)
                invertTree(root->right);
        }
        return root;
    }
    
};

class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root==NULL)
        {
           return root;
        }
        stack<TreeNode*> stk;
        stk.push(root);
        stk.push(root->left);
        stk.push(root->right);
        while(!stk.empty())
        {
            auto s2 =stk.top();
            stk.pop();
            auto s1 = stk.top();
            stk.pop();
            auto s = stk.top();
            stk.pop();
            s->right = s1;
            s->left = s2;
            if (s1!=NULL)
            {
                stk.push(s1);
                stk.push(s1->left);
                stk.push(s1->right);
            }
            if (s2!=NULL)
            {
                stk.push(s2);
                stk.push(s2->left);
                stk.push(s2->right);
            }
        }
        return root;
    }
};


//258
class Solution {
public:
    int addDigits(int num) {
        if (num == 0)
        {
            return 0;
        }
        int n = num%9;
        return (n ==0 )? 9:n;
    }
};

//492
class Solution {
public:
    vector<int> constructRectangle(int area) {
        int a = static_cast<int>(sqrt(area));
        while(area%a !=0 )
        {
            a--;
        }
        return {area/a,a};
    }
};

//530
class Solution {
public:
    int min_dif = INT_MAX; int less = -1;
    int getMinimumDifference(TreeNode* root) {
        if (root->left != NULL) getMinimumDifference(root->left);
        if (less >=0) min_dif = min(min_dif, root->val - less);
        less = root->val;
        if (root->right != NULL) getMinimumDifference(root->right);
        return min_dif;
    }
};

//167
class Solution {
public:
  
vector<int> twoSum(const vector<int>& numbers, int target) {
	multimap<int, int> maps;
	vector<int> res;
	auto size = numbers.size();
	for (int i = 0; i<size; i++)
	{
		//maps[numbers[i]] = i + 1;
		maps.emplace(numbers[i], i + 1);
	}
	for (int i = 0; i<size; i++)
	{
		if (maps.find(target - numbers[i]) != maps.end())
		{
			if (target != 2 * numbers[i])
			{
				res.push_back(i + 1);
				auto tmp = maps.equal_range(target - numbers[i]);
				res.push_back((tmp.first->second));
				break;
			}
			else if (maps.count(numbers[i]) == 2)
			{
				auto t = maps.equal_range(numbers[i]);
				res.push_back((*(t.first)).second);
				res.push_back((*(++t.first)).second);
				break;
			}
		}

	}
	return res;
}

};
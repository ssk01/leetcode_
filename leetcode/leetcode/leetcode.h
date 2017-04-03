#pragma once
#pragma once
#include <iostream>
#include <vector>
#include <utility>
#include <tuple>
#include <memory>
#include <map>
#include <unordered_map>
#include <queue>
#include <string>
#include <stack>
#include <unordered_set>
using namespace std;


struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}

};


//506
//
//using p = pair<int, int>;
//auto comp = [](p a1, p a2) {return a1.first < a2.first; };
//vector<string> findRelativeRanks(vector<int>& nums) {
//	priority_queue<p,vector<p>,decltype(comp)> pq(comp);
//	auto size = nums.size();
//	for (size_t i = 0; i < size; i++)
//	{
//		pq.emplace(nums[i], i);
//	}
//	pq.top();
//	vector<string> res(size, "");
//	int count = 1;
//	for (size_t i = 0; i < size; i++)
//	{
//		if (count == 1)
//		{
//			res[pq.top().second] = "Gold Medal";
//		}
//		else if (count == 2)
//		{
//			res[pq.top().second] = "Silver Medal";
//		}
//		else if (count == 3)
//		{
//			res[pq.top().second] = "Bronze Medal";
//		}
//		else
//		{
//			res[pq.top().second] = to_string(i + 1);
//		}
//		count++;
//		pq.pop();
//	}
//	return res;
//}
//


//
//vector<int> twoSum(const vector<int>& numbers, int target) {
//	multimap<int, int> maps;
//	vector<int> res;
//	auto size = numbers.size();
//	for (int i = 0; i<size; i++)
//	{
//		//maps[numbers[i]] = i + 1;
//		maps.emplace(numbers[i], i + 1);
//	}
//	for (int i = 0; i<size; i++)
//	{
//		if (maps.find(target - numbers[i]) != maps.end())
//		{
//			if (target != 2 * numbers[i])
//			{
//				res.push_back(i + 1);
//				auto tmp = maps.equal_range(target - numbers[i]);
//				res.push_back((tmp.first->second));
//				break;
//			}
//			else if (maps.count(numbers[i]) == 2)
//			{
//				auto t = maps.equal_range(numbers[i]);
//				res.push_back((*(t.first)).second);
//				res.push_back((*(++t.first)).second);
//				break;
//			}
//		}
//
//	}
//	return res;
//}
//


//
////455
//int findContentChildren(vector<int>& g, vector<int>& s) {
//	priority_queue<int> pqG{ g.begin(),g.end() };
//	priority_queue<int> pqS{ s.begin(), s.end() };
//	
//	int result = 0;
//	int topg;
//	int tops;
//	while (!pqS.empty() && !pqG.empty())
//	{
//		topg = pqG.top();
//		tops = pqS.top();
//		if (tops >= topg)
//		{
//			pqG.pop();
//			pqS.pop();
//			result++;
//		}
//		else
//		{
//			pqG.pop();
//		}
//	}
//	return result;
//}
//

//
////453
//int minMoves(vector<int>& nums) {
//	int min = INT_MAX;
//	auto size = nums.size();
//	int sum = 0;
//	for (int i = 0; i < size; i++)
//	{
//		sum += nums[i];
//		min = (min < nums[i]) ? min : nums[i];
//	}
//	return sum - size*min;
//}


//
////404
//int sumOfLeftLeaves(TreeNode* root) {
//
//	if (root == NULL)
//	{
//		return 0;
//	}
//	
//	if (root->left != NULL && root->left->left == NULL && root->left->right == NULL)
//	{
//			return  root->left->val + sumOfLeftLeaves(root->right);
//	}
//	return sumOfLeftLeaves(root->right) + sumOfLeftLeaves(root->left);
//}

//int sumOfLeftLeaves(TreeNode* root)
//{
//	int result = 0;
//	stack<TreeNode*> stk;
//	if (root != NULL)
//	{
//		stk.push(root);
//	}
//
//	while (!stk.empty())
//	{
//		auto top = stk.top();
//		stk.pop();
//		if (top->left != NULL)
//		{
//			if (top->left->left == NULL &&top->left->right == NULL)
//			{
//				result += top->left->val;
//			}
//			else
//			{
//				stk.push(top->left);
//			}
//		}
//		if (top->right != NULL)
//		{
//			if (top->right->left == NULL &&top->right->right == NULL)
//			{
//				
//			}
//			else
//			{
//				stk.push(top->right);
//			}
//		}
//	}
//	return result;
//}

//
//bool canConstruct(string ransomNote, string magazine) {
//	map<char, int> maps;
//	for (auto m : magazine)
//	{
//		maps[m]++;
//	}
//	for (auto r : ransomNote)
//	{
//		if (maps.find(r) == maps.end())
//		{
//			return false;
//		}
//		auto i= --maps[r];
//		
//		if (i < 0)
//		{
//			return false;
//		}
//	}
//	return true;
//}


//int maxProfit(vector<int>& prices) {
//	int res=0;
//	int yesterday = INT_MAX;
//	int size = prices.size();
//	for (int i = 0; i < size; i++)
//	{
//		if (prices[i] > yesterday)
//		{
//			res += prices[i] - yesterday;
//		}
//		yesterday = prices[i];
//	}
//	return res;
//}


//
//int firstUniqChar(string s) {
//fuck wrong
//	vector<int> vec(26,0);
//	for (auto c : s)
//	{
//		vec[c - 'a']++;
//	}
//	auto size = s.size();
//	for (int i =0;i<size;i++)
//	{
//		if (vec[s[i] - 'a'] == 1)
//		{
//			return i;
//		}
//	}
//	
//
//}

//int titleToNumber(string s) {
//	int length = s.length();
//	int res=0;
//	for (int i = 0; i < length; i++)
//	{
//		res =   26*res + s[i] - 'A' + 1;
//	}
//	return res;
//}


//
//// #463
//class Solution {
//public:
//	int islandPerimeter(vector<vector<int>>& grid) {
//		int result = 0;
//		int repeat = 0;
//		auto size1 = grid.size();
//		for (int i = 0; i <size1; i++)
//		{
//			auto size = grid[i].size();
//			for (int j = 0; j < size; j++)
//			{
//				if (grid[i][j] == 1)
//				{
//					result++;
//					if (i != 0 && grid[i - 1][j] == 1)
//						repeat++;
//					if (j != 0 && grid[i][j - 1] == 1)
//						repeat++;
//				}
//			}
//		}
//		return 4 * result - 2 * repeat;
//	}
//
//
//};
//
//
////#292
//class Solution {
//public:
//	bool canWinNim(int n) {
//		if (n % 4 == 0)
//			return false;
//		return true;
//	}
//};
//
////#485
//class Solution {
//public:
//	int findMaxConsecutiveOnes(vector<int>& nums) {
//		int res = 0;
//		int max = 0;
//
//		auto size = nums.size();
//		for (int i = 0; i<size; ++i)
//		{
//			if (nums[i])
//			{
//				res++;
//			}
//			else
//			{
//				if (res>max)
//					max = res;
//				res = 0;
//			}
//		}
//
//		return res>max ? res : max;
//	}
//};
//
////#136
//class Solution {
//public:
//	int singleNumber(vector<int>& nums) {
//		auto size = nums.size();
//		int result = 0;
//		for (auto i = 0; i< size; i++)
//		{
//			result ^= nums[i];
//		}
//		return result;
//	}
//};
//
////#448
//class Solution {
//public:
//	vector<int> findDisappearedNumbers(vector<int>& nums) {
//		int len = nums.size();
//		vector<int> result;
//		int m;
//		for (auto i = 0; i <len; ++i)
//		{
//			m = abs(nums[i]) - 1;
//			nums[m] = -abs(nums[m]);
//		}
//		for (auto i = 0; i<len; ++i)
//		{
//			if (nums[i]>0)
//			{
//				result.push_back(i + 1);
//			}
//		}
//		return result;
//
//	}
//};
//
////520
//bool detectCapitalUse(string word) {
//	bool first = true;
//	bool large = true;
//	bool fuck = true;
//	bool second = true;
//	auto size = word.size();
//	for (int i = 0; i< size; ++i)
//	{
//
//		if (first)
//		{
//			if (word[i] >= 'a')
//			{
//				large = false;
//			}
//			else
//			{
//
//			}
//			first = false;
//		}
//		else
//		{
//			if (!large)
//			{
//				if (word[i] <= 'Z')
//					return false;
//			}
//			else
//			{
//				if (word[i]>'Z')
//				{
//					if (second)
//					{
//						fuck = true;
//						second = false;
//					}
//					else
//					{
//						if (!fuck)
//						{
//							return false;
//						}
//					}
//				}
//				else if (word[i] <= 'Z')
//				{
//					if (second)
//					{
//						fuck = false;
//						second = false;
//					}
//					if (fuck)
//					{
//						return false;
//
//					}
//				}
//			}
//		}
//
//	}
//	return true;
//}
//
////104
//class Solution {
//public:
//	int maxDepth(TreeNode* root) {
//
//		return (root == NULL) ? 0 : max(maxDepth(root->left), maxDepth(root->right)) + 1;
//	}
//};
//class Solution {
//public:
//	int maxDepth(TreeNode* root) {
//		if (root == NULL)
//		{
//			return 0;
//		}
//		queue<TreeNode*> que;
//		int res = 0;
//		que.push(root);
//		while (!que.empty())
//		{
//			res++;
//			for (int i = 0, n = que.size(); i<n; i++)
//			{
//				auto t = que.front();
//				que.pop();
//				if (t->left != NULL)
//				{
//					que.push(t->left);
//				}
//				if (t->right != NULL)
//				{
//					que.push(t->right);
//				}
//			}
//		}
//		return res;
//	}
//};
//
////389
//class Solution {
//public:
//	char findTheDifference(string s, string t) {
//		int size = s.size();
//		int a = 0; int b = 0;
//		for (int i = 0; i<size; i++)
//		{
//			a += (int)s[i];
//			b += (int)t[i];
//		}
//		b += (int)t.back();
//		return (char)(b - a);
//	}
//};
//
////226
//class Solution {
//public:
//	TreeNode* invertTree(TreeNode* root) {
//		if (root != NULL)
//		{
//			TreeNode * tmp = root->left;
//			root->left = root->right;
//			root->right = tmp;
//			if (root->left != NULL)
//				invertTree(root->left);
//			if (root->right != NULL)
//				invertTree(root->right);
//		}
//		return root;
//	}
//
//};
//
//class Solution {
//public:
//	TreeNode* invertTree(TreeNode* root) {
//		if (root == NULL)
//		{
//			return root;
//		}
//		stack<TreeNode*> stk;
//		stk.push(root);
//		stk.push(root->left);
//		stk.push(root->right);
//		while (!stk.empty())
//		{
//			auto s2 = stk.top();
//			stk.pop();
//			auto s1 = stk.top();
//			stk.pop();
//			auto s = stk.top();
//			stk.pop();
//			s->right = s1;
//			s->left = s2;
//			if (s1 != NULL)
//			{
//				stk.push(s1);
//				stk.push(s1->left);
//				stk.push(s1->right);
//			}
//			if (s2 != NULL)
//			{
//				stk.push(s2);
//				stk.push(s2->left);
//				stk.push(s2->right);
//			}
//		}
//		return root;
//	}
//};
//
//
////258
//class Solution {
//public:
//	int addDigits(int num) {
//		if (num == 0)
//		{
//			return 0;
//		}
//		int n = num % 9;
//		return (n == 0) ? 9 : n;
//	}
//};
//
////492
//class Solution {
//public:
//	vector<int> constructRectangle(int area) {
//		int a = static_cast<int>(sqrt(area));
//		while (area%a != 0)
//		{
//			a--;
//		}
//		return{ area / a,a };
//	}
//};
//
////530
//class Solution {
//public:
//	int min_dif = INT_MAX; int less = -1;
//	int getMinimumDifference(TreeNode* root) {
//		if (root->left != NULL) getMinimumDifference(root->left);
//		if (less >= 0) min_dif = min(min_dif, root->val - less);
//		less = root->val;
//		if (root->right != NULL) getMinimumDifference(root->right);
//		return min_dif;
//	}
//};
//
////167
//class Solution {
//public:
//
//	vector<int> twoSum(const vector<int>& numbers, int target) {
//		multimap<int, int> maps;
//		vector<int> res;
//		auto size = numbers.size();
//		for (int i = 0; i<size; i++)
//		{
//			//maps[numbers[i]] = i + 1;
//			maps.emplace(numbers[i], i + 1);
//		}
//		for (int i = 0; i<size; i++)
//		{
//			if (maps.find(target - numbers[i]) != maps.end())
//			{
//				if (target != 2 * numbers[i])
//				{
//					res.push_back(i + 1);
//					auto tmp = maps.equal_range(target - numbers[i]);
//					res.push_back((tmp.first->second));
//					break;
//				}
//				else if (maps.count(numbers[i]) == 2)
//				{
//					auto t = maps.equal_range(numbers[i]);
//					res.push_back((*(t.first)).second);
//					res.push_back((*(++t.first)).second);
//					break;
//				}
//			}
//
//		}
//		return res;
//	}
//
//};
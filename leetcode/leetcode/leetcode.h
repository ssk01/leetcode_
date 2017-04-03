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
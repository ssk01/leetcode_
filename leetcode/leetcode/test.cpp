#include "leetcode.h"
template<typename t>
ostream& operator<<( ostream& v, vector<t> a)
{
	for (auto &b : a)
	{
		v << b << "  ";
	}
	v << endl;
	return v;
}

int main()
{
	/*vector<int> a{ 3,2,4};
	vector<int> b{ 2,3};

	cout << (-11) / 10 << endl;
	cout << (-11) % 10 << endl;*/
	//cout << myAtoi("1") << endl;
	vector<int> s{ 2,3,7 };
	auto res =combinationSum(s, 7);

}



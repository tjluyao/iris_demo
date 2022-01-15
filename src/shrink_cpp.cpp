#include <vector>
#include <limits>
#include <utility>
#include "shrink_cpp.h"
using namespace std;

std::vector<std::vector<int> > partition(std::vector<std::vector<float> > keys, std::vector<int> nss, std::vector<std::vector<int> > cnts)
{
    vector<vector<int> > rets;
	for (int d = 0; d < keys.size(); d++)
	{
	    vector<int> ret;
	    for (int i = 0; i < keys[d].size(); i++)
	        ret.push_back(i);
		
		std::vector<float> valtup;
		std::vector<float> valdif;
		
		for (int i = 0; i < keys[d].size() - 1; i++)
		{
			float vd = keys[d][i + 1] - keys[d][i];
			if(vd==0) vd = 1;
			valtup.push_back((cnts[d][i + 1] + cnts[d][i]) * vd);
			valdif.push_back(vd);
		}
		
		while (keys[d].size() > nss[d])
		{
			float mintup = *min_element(valtup.begin(), valtup.end());
			vector<int> ids;
			for (int i = 0; i < valtup.size(); i++)
				if(valtup[i]==mintup)
					ids.push_back(i);
			float mindif = std::numeric_limits<float>::max();
			for (int i = 0; i < ids.size(); i++)
				mindif = min(mindif, valdif[ids[i]]);
			vector<int> minvds;
			for (int i = 0; i < ids.size(); i++)
				if(valdif[ids[i]]==mindif)
					minvds.push_back(i);
			int minvd = ids[minvds[minvds.size() / 2]];
			//cout<<"-----valtup:"<<valtup[0]<<","<<valtup[1]<<",valdif:"<<valdif[0]<<","<<valdif[1]<<",minvd:"<<minvd<<endl;
			
			cnts[d][minvd + 1] = cnts[d][minvd] + cnts[d][minvd + 1];
			keys[d].erase(keys[d].begin() + minvd);
			ret.erase(ret.begin() + minvd);
			cnts[d].erase(cnts[d].begin() + minvd);
			valtup.erase(valtup.begin() + minvd);
			valdif.erase(valdif.begin() + minvd);
			
			if (minvd > 0)
			{
				float vd = keys[d][minvd] - keys[d][minvd - 1];
				if(vd==0) vd = 1;
				valtup[minvd - 1] = (cnts[d][minvd - 1] + cnts[d][minvd]) * vd;
				valdif[minvd - 1] = vd;
			}
			if (minvd < valtup.size() - 1)
			{
				float vd = keys[d][minvd + 1] - keys[d][minvd];
				if(vd==0) vd = 1;
				valtup[minvd] = (cnts[d][minvd] + cnts[d][minvd + 1]) * vd;
				valdif[minvd] = vd;
			}
		}
		rets.push_back(ret);
	}
	return rets;
}
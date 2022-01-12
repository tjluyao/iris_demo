#include "cords.h"

static vector<int> primes{ 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431 };

static int nbuckets = 10;
static float keyratio = 0.85;

class choice
{
public:
	vector<int> id;
	string ch = "";
	int dvs = 0;
	float corr = 0;
	choice(vector<int> id)
	{
		sort(id.begin(), id.end());
		this->id = id;
	}
	choice(vector<int> id, int dvs, string choice, float corr)
	{
		sort(id.begin(), id.end());
		this->id = id;
		this->dvs = dvs;
		this->ch = choice;
		this->corr = corr;
	}
};

void cords2d(vector<vector<string>> tab, vector<choice>& ch, int n_rows, map<int, map<string, float>> p, map<int, vector<string>> d)
{
	//#pragma omp parallel for
	for (int t = 0; t < ch.size(); t++)
	{
		int i = ch[t].id[0];
		int j = ch[t].id[1];
		double phi = 0;
		auto di = d[i], dj = d[j];
		auto pi = p[i], pj = p[j];
		int sdi = di.size(), sdj = dj.size();

		vector<int> dvs;
		for (int r = 0; r < n_rows; r++)
			dvs.push_back(hash<string>()(tab[r][i]) * primes[0] + hash<string>()(tab[r][j]) * primes[1]);
		sort(dvs.begin(), dvs.end());
		dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

		if (di.size() <= 1 || dj.size() <= 1)
		{
			ch[t] = choice(vector<int>{i, j}, dvs.size(), "AVI", 0.0);
			continue;
		}

		//trim key columns
		if (di.size() >= n_rows * 0.8 || dj.size() >= n_rows * 0.8)
		{
			ch[t] = choice(vector<int>{i, j}, dvs.size(), "AVI", phi);
			continue;
		}

		//bucketing
		if (dvs.size() > 225)
		{
			vector<string> dtmp;
			map<string, float> ptmp;
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(di[n * di.size() / nbuckets]);
				ptmp[di[n * di.size() / nbuckets]] = 0;
				for (int t = n * di.size() / nbuckets; t < (n + 1) * di.size() / nbuckets; t++)
					ptmp[di[n * di.size() / nbuckets]] += pi[di[t]];
			}
			pi = ptmp;
			dtmp.push_back(di[di.size() - 1]);
			di = vector<string>(dtmp);
			dtmp.clear();
			ptmp.clear();
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(dj[n * dj.size() / nbuckets]);
				ptmp[dj[n * dj.size() / nbuckets]] = 0;
				for (int t = n * dj.size() / nbuckets; t < (n + 1) * dj.size() / nbuckets; t++)
					ptmp[dj[n * dj.size() / nbuckets]] += pj[dj[t]];
			}
			pj = ptmp;
			dtmp.push_back(dj[dj.size() - 1]);
			dj = vector<string>(dtmp);
		}

		for (int ni = 0; ni < di.size() - 1; ni++)
		{
			for (int nj = 0; nj < dj.size() - 1; nj++)
			{
				float pij = 0;
				for (int r = 0; r < n_rows; r++)
				{
					if (tab[r][i] >= di[ni] && tab[r][j] >= dj[nj] && tab[r][i] < di[ni + 1] && tab[r][j] < dj[nj + 1])
						pij++;
				}
				pij /= n_rows;
				phi += pow(pij - pi[di[ni]] * pj[dj[nj]], 2) / (pi[di[ni]] * pj[dj[nj]]);
			}
		}
		phi /= (min(di.size(), dj.size()) - 1);


		if (phi < 0.003)
			ch[t] = choice(vector<int>{i, j}, dvs.size(), "AVI", phi);
		else
		{
			if (dvs.size() < 225)
				ch[t] = choice(vector<int>{i, j}, dvs.size(), "Sparse", phi);
			else if (min(sdi, sdj) < 15)
				ch[t] = choice(vector<int>{i, j}, dvs.size(), "Hist", phi);
			else
				ch[t] = choice(vector<int>{i, j}, dvs.size(), "Iris", phi);
		}
	}
}

void cords3d(vector<vector<string>> tab, vector<choice>& ch, int n_rows, map<int, map<string, float>> p, map<int, vector<string>> d)
{
	//#pragma omp parallel for
	for (int t = 0; t < ch.size(); t++)
	{
		int i = ch[t].id[0];
		int j = ch[t].id[1];
		int k = ch[t].id[2];
		double phi = 0;
		auto di = d[i], dj = d[j], dk = d[k];
		auto pi = p[i], pj = p[j], pk = p[k];
		int sdi = di.size(), sdj = dj.size(), sdk = dk.size();

		vector<int> dvs;
		for (int r = 0; r < n_rows; r++)
			dvs.push_back(hash<string>()(tab[r][i]) * primes[0] + hash<string>()(tab[r][j]) * primes[1] + hash<string>()(tab[r][k]) * primes[2]);
		sort(dvs.begin(), dvs.end());
		dvs.erase(unique(dvs.begin(), dvs.end()), dvs.end());

		//bucketing
		if (dvs.size() > 150)
		{
			vector<string> dtmp;
			map<string, float> ptmp;
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(di[n * di.size() / nbuckets]);
				ptmp[di[n * di.size() / nbuckets]] = 0;
				for (int t = n * di.size() / nbuckets; t < (n + 1) * di.size() / nbuckets; t++)
					ptmp[di[n * di.size() / nbuckets]] += pi[di[t]];
			}
			pi = ptmp;
			dtmp.push_back(di[di.size() - 1]);
			di = vector<string>(dtmp);
			dtmp.clear();
			ptmp.clear();
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(dj[n * dj.size() / nbuckets]);
				ptmp[dj[n * dj.size() / nbuckets]] = 0;
				for (int t = n * dj.size() / nbuckets; t < (n + 1) * dj.size() / nbuckets; t++)
					ptmp[dj[n * dj.size() / nbuckets]] += pj[dj[t]];
			}
			pj = ptmp;
			dtmp.push_back(dj[dj.size() - 1]);
			dj = vector<string>(dtmp);
			dtmp.clear();
			ptmp.clear();
			for (int n = 0; n < nbuckets; n++)
			{
				dtmp.push_back(dk[n * dk.size() / nbuckets]);
				ptmp[dk[n * dk.size() / nbuckets]] = 0;
				for (int t = n * dk.size() / nbuckets; t < (n + 1) * dk.size() / nbuckets; t++)
					ptmp[dk[n * dk.size() / nbuckets]] += pk[dk[t]];
			}
			pk = ptmp;
			dtmp.push_back(dk[dk.size() - 1]);
			dk = vector<string>(dtmp);
		}

		for (int ni = 0; ni < di.size() - 1; ni++)
		{
			for (int nj = 0; nj < dj.size() - 1; nj++)
			{
				for (int nk = 0; nk < dk.size() - 1; nk++)
				{
					float pijk = 0;
					for (int r = 0; r < n_rows; r++)
					{
						if (tab[r][i] >= di[ni] && tab[r][j] >= dj[nj] && tab[r][k] >= dk[nk]
							&& tab[r][i] < di[ni + 1] && tab[r][j] < dj[nj + 1] && tab[r][k] < dk[nk + 1])
							pijk++;
					}
					pijk /= n_rows;
					phi += pow(pijk - pi[di[ni]] * pj[dj[nj]] * pk[dk[nk]], 2) / (pi[di[ni]] * pj[dj[nj]] * pk[dk[nk]]);
				}
			}
		}
		phi /= min(min(di.size() * dj.size(), dj.size() * dk.size()), di.size() * dk.size()) - 1;

		if (phi < 0.003)
			ch[t] = choice(vector<int>{i, j, k}, dvs.size(), "AVI", phi);
		else
		{
			if (dvs.size() < 150)
				ch[t] = choice(vector<int>{i, j, k}, dvs.size(), "Sparse", phi);
			else if (min(min(sdi, sdj), sdk) < 15)
				ch[t] = choice(vector<int>{i, j, k}, dvs.size(), "Hist", phi);
			else
				ch[t] = choice(vector<int>{i, j, k}, dvs.size(), "Iris", phi);
		}
	}
}

void CORDS(string fnm, string scols)
{
	ofstream ofs("CORDS.log");
	vector<vector<string>> tab;
	string line, v;
	ifstream in(fnm.c_str());
	while (getline(in, line))
	{
		vector<string> ln;
		istringstream ss(line);
		int id = 0;
		while (getline(ss, v, '|'))
			ln.push_back(v);
		tab.push_back(ln);
	}
	vector<int> cols;
	istringstream ss(scols);
	int id = 0;
	while (getline(ss, v, ','))
		cols.push_back(stoi(v));

	vector<choice> ch;
	for (int i = 0; i < cols.size(); i++)
	{
		for (int j = i + 1; j < cols.size(); j++)
		{
			ch.push_back(choice(vector<int>{cols[i], cols[j]}));
		}
	}
	map<int, map<string, float>> p;
	map<int, vector<string>> d;
	int n_rows = tab.size();
	int n_cols = tab[0].size();

	for (int t = 0; t < cols.size(); t++)
	{
		map<string, float> pi;
		vector<string> di;
		for (int j = 0; j < n_rows; j++)
			di.push_back(tab[j][cols[t]]);
		sort(di.begin(), di.end());
		di.erase(unique(di.begin(), di.end()), di.end());
		for (int n = 0; n < di.size(); n++)
		{
			pi[di[n]] = 0;
			for (int j = 0; j < n_rows; j++)
			{
				if (tab[j][cols[t]] == di[n])
					pi[di[n]]++;
			}
			pi[di[n]] /= n_rows;
			p[cols[t]] = pi;
			d[cols[t]] = di;
		}
	}

	cords2d(tab, ch, n_rows, p, d);
	sort(ch.begin(), ch.end(), [](const choice a, const choice b) {return a.corr > b.corr; });

	//output
	vector<int> ds1;
	ofs << cols.size() - ds1.size() << endl;
	for (int i = 0; i < cols.size(); i++)
	{
		if (find(ds1.begin(), ds1.end(), cols[i]) == ds1.end())
		    if(d[cols[i]].size() < n_rows * keyratio)
			    ofs << cols[i] << "\t" << d[cols[i]].size() << "\tAVI" << endl;
			else
			    ofs << cols[i] << "\t" << d[cols[i]].size() << "\tKey" << endl;
	}
	ofs << ch.size() << endl;
	for (auto it = ch.begin(); it != ch.end(); it++)
		ofs << it->id[0] << "\t" << it->id[1] << "\t" << it->dvs << "\t" << it->corr << "\t" << it->ch << endl;
	ofs.flush();
	ofs.close();
}
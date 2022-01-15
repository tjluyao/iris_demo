#include "cords.h"

static vector<int> primes{ 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431 };

static int nbuckets = 16;
static float keyratio_1d = 0.85;
static float keyratio_2d = 0.75;

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

        //prune cases
        if ((di.size () >= n_rows * keyratio_1d || dj.size() >= n_rows * keyratio_1d) ||
		    (di.size() >= n_rows * keyratio_2d && dj.size() >= n_rows * keyratio_2d && dvs.size() >= n_rows * keyratio_2d))
		{
			ch[t] = choice(vector<int>{i, j}, dvs.size(), "AVI", -1, vector<int>{sdi, sdj});
			continue;
		}

        //conservative - more sparse depending on storage budget
        if(dvs.size() <= 64)
        {
            ch[t] = choice(vector<int>{i, j}, dvs.size(), "Sparse", 1, vector<int>{sdi, sdj});
            continue;
        }

		bool ishist = (max(sdi/sdj, sdj/sdi) >= 128);
        if(ishist)
        {
            ch[t] = choice(vector<int>{i, j}, dvs.size(), "Hist", 0, vector<int>{sdi, sdj});
            continue;
        }

		ch[t] = choice(vector<int>{i, j}, dvs.size(), "-", phi, vector<int>{sdi, sdj});

		//bucketing
		if (dvs.size() > nbuckets * nbuckets)
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
			
			//di.erase(unique(di.begin(), di.end()), di.end());			
			//dj.erase(unique(dj.begin(), dj.end()), dj.end());
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

        ch[t].corr = phi;
        if(ishist) ch[t].ch = "Hist";
		if (phi < 0.001)
		{
            ch[t].ch = "AVI";
            ch[t].corr = -1;
		}
	}
}

void CORDS(string ifnm, string ofnm, string scols)
{
	ofstream ofs(ofnm.c_str());
	vector<vector<string>> tab;
	string line, v;
	ifstream in(ifnm.c_str());
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
		//ofs << cols[t] << ": " << di.size() << endl;
	}

	cords2d(tab, ch, n_rows, p, d);
	sort(ch.begin(), ch.end(), [](const choice a, const choice b) {return a.corr > b.corr; });

	//output
	ofs << cols.size() << endl;
	for (int i = 0; i < cols.size(); i++)
	{
        if(d[cols[i]].size() < n_rows * keyratio_1d)
            ofs << cols[i] << "\t" << d[cols[i]].size() << "\tAVI" << endl;
        else
            ofs << cols[i] << "\t" << d[cols[i]].size() << "\tKey" << endl;
	}
	ofs << ch.size() << endl;
	for (auto it = ch.begin(); it != ch.end(); it++)
		ofs << it->id[0] << "\t" << it->id[1] << "\t" << it->dvs << "\t" << it->corr << "\t" << it->ch << "\t" << it->dv[0] << "\t" << it->dv[1] << endl;
	ofs << "0" << endl;
	ofs.flush();
	ofs.close();
}

int main()
{
	string ifnm = "/iris/dataset_public/DMV/sample";
	string ofnm = "CORDS.log";
	string cols = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19";

	CORDS(ifnm, ofnm, cols);
}


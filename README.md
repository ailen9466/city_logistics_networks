# city_logistics_networks
This project contains the code and network data for the paper "Community structure and resilience of the city logistics networks in China". The explanation for the code part is for the community detection algorithm, specifically the myfun_algo_city_cluster function in community_detection_algo.py (with other functions serving auxiliary roles).

To get the result, simply run the community_detection_algo.py, and result r1 is the communities detailed, r2 is the number of cities' community. 

File community_detection_algo.py is the community detection algorithm file. 

File data_city_neighbor.pkl is the binary file of two variable, one is city_neighbor, a dictionary, shows the neighbor relation of the cities, the other one is the cities' code, which could be searched online with the name "administrative division codes of the people's republic of china. URL: https://en.wikipedia.org/wiki/Administrative_division_codes_of_the_People%27s_Republic_of_China .

File data_network.csv is the network data, who has 3 columns: from_city, to_city, and normalized link weight, which represent the departure, the destination, and the normalized link weight respectively.

It's worth noting that since the data cannot be directly published online, users will need to contact the authors if they have further requests.

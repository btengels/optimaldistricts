#!/bin/bash
# A short script that uses mapshaper to simplify the .shp files
for d in * ; do
<<<<<<< HEAD



	# mapshaper ./$d/*final.shp -simplify 10% keep-shapes -o ./$d/precinct/precinct.shp
	# mapshaper ./$d/congress_geo.shp -simplify 50% keep-shapes -o ./$d/congress_dist/congress.shp
	# mapshaper ./$d/tract_geo.shp -simplify 30% keep-shapes -o ./$d/census_tract/census.shp	
	
	# # delete duplicate files, if any 
	rm $d/*.cpg
	rm $d/*.shx
	rm $d/*.sbx
	rm $d/*.sbn
	rm $d/*.prj
	rm $d/*.dbf
	rm $d/*.shp

	# rm $d/precinct/precinct*.shp
	# rm $d/precinct/precinct*.shx
	# rm $d/precinct/precinct*.prj
	# rm $d/precinct/precinct*.dbf	

	# rm $d/congress_dist/congress*.shp
	# rm $d/congress_dist/congress*.shx
	# rm $d/congress_dist/congress*.prj
	# rm $d/congress_dist/congress*.dbf		

	# rm $d/census_tract/census*.shp
	# rm $d/census_tract/census*.shx
	# rm $d/census_tract/census*.prj
	# rm $d/census_tract/census*.dbf	

	# mapshaper ./$d/*final.shp -simplify 10% keep-shapes -o ./$d/precinct/precinct.shp
	# mapshaper ./$d/congress_geo.shp -simplify 50% keep-shapes -o ./$d/congress_dist/congress.shp
	# mapshaper ./$d/tract_geo.shp -simplify 30% keep-shapes -o ./$d/census_tract/census.shp	

=======
	mapshaper -i ./$d/*.shp -simplify .2 -o ./$d/$d-simple.shp	
>>>>>>> 5e28414894bb50e3545acdb37c407071b975a515
done


# mapshaper ./WA/WA_final.shp -simplify 10% keep-shapes -o ./WA/simple/simple.shp	
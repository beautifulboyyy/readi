from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://127.0.0.1:3001/sparql")
sparql.setQuery("""
    SELECT ?x WHERE {
        ?x <http://rdf.freebase.com/ns/type.object.type> <http://rdf.freebase.com/ns/common.topic> .
    } LIMIT 10
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result["x"]["value"])

# deploy-cc-classification



### cURL
`curl -X POST "https://8000-01japfr01tfg2q9519th36eftb.cloudspaces.litng.ai/" ^
-H "Authorization: Bearer cc-classify" ^
-F "request=@ ..." ^
-k
`
`curl --request GET \
  --url https://8000-01japfr01tfg2q9519th36eftb.cloudspaces.litng.ai/ \
  --header 'Authorization: Bearer cc-classify'
`

python -m unittest test/test.py 

python -m test.test_batch image_test/61f0efa8cec542a39807b481.png image_test/61f01c5fcec542a39807a20c.png       

python -m test.test_load.py --image_dir image_test --num-requests 10   

python -m test.test_stress --image_dir="image_test" --initial-requests=10 --max-requests=20 --step=10 --cycles=1

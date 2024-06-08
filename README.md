1- telechargi l dataset w hot l folder fel projet

2- telechargi l model generatif bech testaamlou local https://cdn-lfs.huggingface.co/repos/9a/99/9a993d952a30eabe91c45846982e8532f96d6ca9f06835091d647be7c6367ca6/8799ff9763de13d7f30a683d653018e114ed24a6a819667da4f5ee10f9e805fe?response-content-disposition=inline%3B+filename*%3DUTF-8%27%27kcv_diffusion_model.h5%3B+filename%3D%22kcv_diffusion_model.h5%22%3B&Expires=1718032071&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODAzMjA3MX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy5odWdnaW5nZmFjZS5jby9yZXBvcy85YS85OS85YTk5M2Q5NTJhMzBlYWJlOTFjNDU4NDY5ODJlODUzMmY5NmQ2Y2E5ZjA2ODM1MDkxZDY0N2JlN2M2MzY3Y2E2Lzg3OTlmZjk3NjNkZTEzZDdmMzBhNjgzZDY1MzAxOGUxMTRlZDI0YTZhODE5NjY3ZGE0ZjVlZTEwZjllODA1ZmU%7EcmVzcG9uc2UtY29udGVudC1kaXNwb3NpdGlvbj0qIn1dfQ__&Signature=uTj71tXIWs92I8ykt-mxiAQlqEH4ittyhjlzK6igGYN-am60suQ6BNA4rJp9ecavQzLu7vu6-XGqX8xSwYUQ5Y74ZGkeusJBv7R3be5Hyzcm6jpfuRtyxH1DHWqou0W9eDSw2zE%7Ew8gb9HI8k6X2X3Xcb8veU96XxWHiN4teMODg7RTwfaYonxCSSxt2eEFKDurI%7Ef%7ET%7EcxXGn5yOuD0iaitGotYFdoXuXI2jmTi0ZBSxjdsXCcVh%7E95eO8MYB4ePD12l95ZDhkejQfXfNee3h7sIsqjbQ8XE%7Ey6XqdGeI%7EnV0F%7E3JvLc2AKE8G1MhnIyo3i0VVDyZxGaYDy%7EGir0Q__&Key-Pair-Id=KVTP0A1DKRTAX   w hotou f folder models


3- l environnement : python3.11.7, create virtual env w installi : 
python -m venv venv (fel shell)
cd venv (fel cmd)
cd Scripts 
activate
pip install -r requirements.txt


4- run setup.py 
5- run app.py

ki theb etesti tchouf l output, sob postman wala thunder wala eli theb 
f postman : 

- l modele generatif : 
new request, hotha POST, colli http://127.0.0.1:5000/generate , fel Headers hott Key : Content-Type ; Value : application/json , fel Body selecti raw, thabet menha JSON al imin w hot l prompt eli theb aalih exemple : 
{
  "prompt": "a monkey holding a gun"
}

- l class_model : 
new request, POST, http://127.0.0.1:5000/classify , fel Headers hott Key : Content-Type ; Value : application/json , f Body selecti form-data w hot :
Key : file wl type File  ; Value : uploadi l image prompt       


fel les 2 cas thez l response tcopiha w temchi l encode_img.py w thot eli copitou f reponse_data w t'executi ( tsawer l class_model ijiw fl folder similiar_images w l generatif ijiwek f uploads)
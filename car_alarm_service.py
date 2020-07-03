import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

app = FastAPI()


@app.get("/")
async  def root():
    return 'Hello World!'


@app.post("/car_alarm")
async def car_alarm(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
        return item_dict


if __name__ == '__main__':
    uvicorn.run(app)
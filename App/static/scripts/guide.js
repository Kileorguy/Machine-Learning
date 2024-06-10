window.onload = () => {



    const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','Space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    let container = document.getElementById('guideContainer')
    let prev = document.getElementById('prevBtn')
    let next = document.getElementById('nextBtn')
    let page = 0
    let min = 0
    let max = 4
    let total = 27
    let cards = []

    let removeCards = () => {
        for (let i = page*6;i < (page+1)*6;i++){
            if (i < total)
                container.removeChild(cards[i])
        }
    }

    let addCards = () => {
        for (let i = page*6;i < (page+1)*6;i++){
            if (i < total)
                container.appendChild(cards[i])
        }
    }

    let initCard = () => {
        
        for (let i = 0;i < total;i++){
            let card = document.createElement('div')
            let image = document.createElement('img')
            let text = document.createElement('h4')
            card.className='card'
            text.innerHTML = classes[i]
            text.className = 'cardText'
            image.src = `./static/asset/${classes[i]}.jpg`
            image.className = 'cardImage'
            card.append(image, text)
            cards.push(card)
        }
    }

    initCard()
    addCards()
    
    prev.addEventListener('click', () => {
        if (page > min){
            removeCards()
            page--
            addCards()
        }
    })
    next.addEventListener('click', () => {
        if (page < max){
            removeCards()
            page++
            addCards()
        }
    })
}
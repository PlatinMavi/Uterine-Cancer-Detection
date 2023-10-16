const overlay = document.querySelector('.modal-overlay')
overlay.addEventListener('click', toggleModal)

var closemodal = document.querySelectorAll('.modal-close')
for (var i = 0; i < closemodal.length; i++) {
    closemodal[i].addEventListener('click', toggleModal)
}

document.onkeydown = function(evt) {
    evt = evt || window.event
    var isEscape = false
    if ("key" in evt) {
    isEscape = (evt.key === "Escape" || evt.key === "Esc")
    } else {
    isEscape = (evt.keyCode === 27)
    }
    if (isEscape && document.body.classList.contains('modal-active')) {
    toggleModal()
    }
};


function toggleModal () {
    const body = document.querySelector('body')
    const modal = document.querySelector('.modal')
    modal.classList.toggle('opacity-0')
    modal.classList.toggle('pointer-events-none')
    body.classList.toggle('modal-active')
}

function addLoader(elementId) {
    const html = `
      <div class="flex items-center justify-center h-96">
          <div class="relative">
              <div class="h-24 w-24 rounded-full border-t-8 border-b-8 border-gray-200"></div>
              <div class="absolute top-0 left-0 h-24 w-24 rounded-full border-t-8 border-b-8 border-emerald-400 animate-spin"></div>
              <p class="text-2xl -translate-x-3 mt-2 text-center w-max font-thin">Yükleniyor</p>
          </div>
          
      </div>
    `;
  
    const element = document.getElementById(elementId);
    if (element) {
      element.innerHTML = html;
    } else {
      console.error(`Element with ID ${elementId} not found.`);
    }
}

function addResults(elementId,r1,r2,r3) {
    let state = undefined
    if (r1 === 1){
        state = "Kanser"
    }else{
        state = "Sağlıklı"
    }
    const html = `
      <div class="flex items-center justify-center h-96">
        <div class="text-center text-xl">
            <h3>Tahmin Durumu: ${state}</h3>
            <h3>Sağlıklı Oy Algoritma Değeri: ${r2}</h3>
            <h3>Kanser Oy Algoritma Değeri: ${r3}</h3>
        </div>
          
      </div>
    `;
  
    const element = document.getElementById(elementId);
    if (element) {
      element.innerHTML = html;
    } else {
      console.error(`Element with ID ${elementId} not found.`);
    }
}
  

async function predict(){
    toggleModal()
    addLoader("body-modal")

    const WBC = document.getElementById("WBC").value
    const NEU = document.getElementById("NEU").value
    const LYM = document.getElementById("LYM").value
    const MONO = document.getElementById("MONO").value

    const EOS = document.getElementById("EOS").value
    const BASO = document.getElementById("BASO").value
    const RBC = document.getElementById("RBC").value
    const HGB = document.getElementById("HGB").value

    const HCT = document.getElementById("HCT").value
    const MCV = document.getElementById("MCV").value
    const MCH = document.getElementById("MCH").value
    const MCHC = document.getElementById("MCHC").value

    const RDWSD = document.getElementById("RDWSD").value
    const RDWCV = document.getElementById("RDWCV").value
    const PLT = document.getElementById("PLT").value
    const MPV = document.getElementById("MPV").value

    const PCT = document.getElementById("PCT").value
    const PDW = document.getElementById("PDW").value
    const NRBC = document.getElementById("NRBC").value

    const response = await fetch("/api/prediction", {
        method: "POST",
        mode: "cors",
        headers: {
            "Content-Type": "application/json",
        },
        body:JSON.stringify({
            WBC: WBC,
            NEU: NEU,
            LYM: LYM,
            MONO: MONO,
            EOS: EOS,
            BASO: BASO,
            RBC: RBC,
            HGB: HGB,
            HCT: HCT,
            MCV: MCV,
            MCH: MCH,
            MCHC: MCHC,
            RDWSD: RDWSD,
            RDWCV: RDWCV,
            PLT: PLT,
            MPV: MPV,
            PCT: PCT,
            PDW: PDW,
            NRBC: NRBC,
        }),
    });
    
    const data = await response.json()

    const prediction = data.p
    const voteValue0 = data.v0
    const voteValue1 = data.v1

    addResults("body-modal",prediction,voteValue0,voteValue1)
}
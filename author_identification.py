# -*- coding: utf-8 -*-
# Yazar Tespiti - COME 448 Proje Soru 6

# Imports
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)

# Tekrarlanabilirlik / Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print('Tum kutuphaneler hazir / All libraries ready')
print(f'PyTorch surumu: {torch.__version__}')
print(f'GPU mevcut mu? / GPU available: {torch.cuda.is_available()}')

# Yazar 0: Omer Seyfettin (1884-1920)
# Sade Turkce, askeri/koy/tarih temalari
OMER_SEYFETTIN = [
    'Cephe gerisinde topların gürültüsü hiç dinmiyordu. Askerler siperde sırtlarını duvara yaslamış, sıralarını bekliyorlardı. Hiçbiri korku göstermiyordu, hepsi kararlıydı.',
    'Düşman mevzilerine gece yarısı baskın yapmaya karar verdik. Komutan emri verdiğinde herkes ayağa kalktı. Sessizce, gölge gibi ilerlemeye başladık.',
    'Süngü hücumu emri geldiğinde askerler bir an duraladı. Sonra komutanın sesi yükseldi, hepsi siperden fırladı. Düşman saflarına bir saniyede ulaştılar.',
    'Top mermisi yakınımıza düştü, toprak yağmur gibi üzerimize yağdı. Yanımdaki arkadaşım hiç istifini bozmadı. Sadece başlığını silkti, gülümsedi.',
    'Cephede o gece çok soğuktu. Askerler birbirlerine sokulup sırtlarını ısıtmaya çalışıyordu. Sabah olduğunda bazıları kıpırdamıyordu, donmuşlardı.',
    'Subay tüfeğini omuzladı, kararlı adımlarla ileri atıldı. Askerleri tereddütsüz onu takip etti. Vatan için canlarını vermeye hazırdılar.',
    'Düşman beyaz bayrak çekti, ama biz hâlâ tetikteydik. Komutan ihtiyatlı olmamızı söyledi. Bu hilelere yıllardır şahit olmuştu.',
    'Mermiler başımızın üzerinden ıslık çalarak geçiyordu. Hiç kimse korkmuyordu artık. Birkaç ay önce gelseler kaçışırdık, şimdi alıştık.',
    'Yaralı asker kanlar içinde yere düştü. Yanındaki arkadaşı hemen üzerine eğildi. "Vatan sağ olsun kardeşim" dedi son nefesini verirken.',
    'Çavuş elindeki haritayı dikkatle inceledi. Düşmanın ne yöne ilerlediğini anlamaya çalışıyordu. Sonra parmağıyla bir noktayı gösterdi: "Burada karşılayacağız."',
    'Top arabaları zorlukla yokuşu çıkıyordu. Atlar yorulmuş, askerler bitkindi. Ama gözlerinde hâlâ bir kararlılık vardı, durmayacaklardı.',
    'Kuşatma kırk gündür sürüyordu. Erzakımız azalmış, içme suyumuz tükenmek üzereydi. Yine de hiçbirimiz teslim olmayı düşünmüyorduk.',
    'Genç asker, ilk savaşına gidiyordu. Annesine veda etmiş, ablalarına sarılmıştı. Şimdi cephede, omuzdaşlarının yanında, ilk düşman ateşini bekliyordu.',
    'Çete reisi atının üstünde dimdik duruyordu. Adamlarına seslendi: "Bu gece pusu kuracağız. Düşman geçerken hepsini biçeceğiz."',
    'Şehit olan arkadaşımızı toprağa verdik. Mezarının başında bir dakika sustuk. Sonra silahlarımızı tekrar omuzladık, savaşa devam ettik.',
    'Köyün muhtarı sabah erkenden meydana indi. Etrafına köylüler toplandı. Yeni gelen ferman okundu, herkes baş eğdi.',
    'Pazar günü kahvede oturuyorduk. İhtiyar Mehmet Ağa eski savaş hikâyelerini anlatıyordu. Bütün gençler büyük bir dikkatle dinliyordu.',
    'Köy mektebinin önünden geçerken çocukların sesini duydum. Hocaefendi onlara alfabe öğretiyordu. Sesleri tâ uzaklardan işitiliyordu.',
    'Hasat zamanı geldi, köylüler tarlalara dağıldı. Kadınlar başaklarını topluyor, erkekler harman dövüyordu. Çocuklar arkalarından koşuşuyordu.',
    'Köy meydanında bir at arabası durdu. İçinden bir tüccar indi. Ahali merakla etrafını sardı, sattığı kumaşları seyretmeye başladı.',
    'Akşam ezanı okunduğunda köy bir anda sessizleşti. Herkes işini bıraktı, camiye yöneldi. Yaşlı imam minarede sesleniyordu.',
    'Bahar gelmiş, ovayı yeşillik kaplamıştı. Çobanlar koyunlarını ot yemeye salmıştı. Köpekler etrafta nöbet tutuyordu.',
    'Köyün berberi Hasan Usta hem tıraş yapar hem nasihat verirdi. Kahveye gelen herkes onun yanından geçerdi. Bilgili bir adamdı.',
    'Düğün hazırlıkları üç gündür sürüyordu. Köyün kadınları kazanlar başında pilav pişiriyordu. Çocuklar etrafta sevinçle koşuşuyordu.',
    'Köy halkı kuraklıktan şikâyetçiydi. Aylardır yağmur düşmemişti, ekinler kuruyordu. Hocaefendi yağmur duasına çıktı, halk arkasından yürüdü.',
    'Akşamüstü köyün ihtiyarları kahveye doluştu. Kahveci Memo onlara bardak bardak çay servis etti. Herkes günün haberlerini birbirine anlatıyordu.',
    'Köy bekçisi gece yarısı sokakta dolaşıyordu. Elinde feneri, belinde sopası vardı. Köpekler havlayınca hemen sesin geldiği yöne koşturdu.',
    'Han önünde kervan durmuştu. Develer kum gibi sıralanmış, kervancılar yere oturmuşlardı. Çay pişirip ekmek yiyeceklerdi.',
    'Köy düğününde davul zurna çalıyordu. Gençler halka olmuş oynuyordu. Yaşlılar bir köşede oturmuş, gençliklerini hatırlıyordu.',
    'Yağmur yağmaya başladığında herkes evine kaçıştı. Yalnız ihtiyar Hasan Dede meydanda kaldı. Şemsiyesini açtı, ağır ağır evine yürüdü.',
    'Padişah divanında oturuyordu. Vezirler etrafında saf tutmuştu. Yeni bir sefere çıkma kararı verilecekti.',
    'Yeniçeri ağası saraya çağrılmıştı. Padişahın huzuruna girdi, secde etti. Verilecek emirleri bekliyordu sabırla.',
    "Edirne'ye gelen ordu şehrin kapısında durdu. Sancak beyi onları karşıladı. Saraya buyur edildiler, akşam yemeği hazırlandı.",
    'Defterdar padişaha kese kese altın takdim etti. Bu yılki haraç bolca toplanmıştı. Devletin kasası dolup taşıyordu.',
    'Tımarlı sipahiler atlarını mahmuzladı. Sefer mevsimi başlamıştı, ordugaha yetişmeleri gerekiyordu. Yolda hızla ilerliyorlardı.',
    'Bizans elçisi sultanın huzuruna çıktı. Hediyelerini takdim etti, barış teklif etti. Padişah uzun uzun düşündü, sonunda kabul etti.',
    'Kapıkulu askerleri saraya doğru yürüyordu. Padişahın yıllık ulufesi dağıtılacaktı. Hepsi neşeyle birbirleriyle şakalaşıyordu.',
    'Tarihte ilk kez böyle bir zafer kazanılmıştı. Düşman ordusu darmadağın olmuş, kalesi düşmüştü. Sultan ganimetin beşte birini hazineye koydu.',
    'Kazaskerler divanda fetva veriyordu. Mesele çok çetindi, günlerdir tartışılıyordu. Sonunda âlim bir hoca çıkıp meseleyi çözdü.',
    'Lala paşa şehzadeye ders veriyordu. Saray odasında baş başaydılar. Devlet idaresinin inceliklerini öğretiyordu yavaşça.',
    "Anadolu Beylerbeyi sefere çağrıldı. Atına bindi, askerlerini topladı. Yolda günlerce ilerleyip İstanbul'a vardı.",
    'Sancak çıkarıldı, mehter takımı çalmaya başladı. Ordu sefere çıkıyordu. Halk yollara dökülmüş, askerleri uğurluyordu.',
    'Sultanın annesi haremde oturuyordu. Etrafında cariyeler vardı. Sarayın işlerini ondan sorardı padişah.',
    'Bostancılar bahçede çalışıyordu. Padişahın gelişine az kalmıştı. Her yeri tertemiz, çiçekleri özenle düzenlenmiş bekliyorlardı.',
    'Şehzade tahta çıkacağı günü bekliyordu. Saray protokolü hazırlanmış, tören günü belirlenmişti. O ise odasında dua ediyordu.',
    'Çavuş otuz yıllık askerdi. Cephe gerisinde gençlere savaş öğretiyordu. Her birinin omuzuna eliyle dokunup nasihat veriyordu.',
    'Mülazım çok gençti, ama tecrübeliydi. İki savaş görmüş, üç madalya almıştı. Askerleri ona güveniyor, sözünden çıkmıyordu.',
    'Onbaşı tabancasını çekti, havaya bir el ateş etti. Birlik anında suspus oldu. Düşman yaklaşıyordu, hazırlanmaları gerekiyordu.',
    'Subay askerlerini bir araya topladı. "Vatan için canınızı vermekten çekinmeyin" dedi sert sesle. "Şehit olmak en büyük şereftir."',
    'Yaşlı bir asker tüfeğini omzundan indirdi. Çelikten beden, kanı kuvvetli adamdı. Otuz yıldır cephe gerisinde dolaşıyordu.',
    'Genç teğmen ilk defa kumanda ediyordu. Heyecanla emirler veriyordu. Askerler onu hayretle ve sevgiyle dinliyordu.',
    'Süvari birliği toz duman içinde geliyordu. Atların nalları yerde gümbürtü çıkarıyordu. Düşman karşılarında darmadağın oluyordu.',
    'Çavuş cephe gerisinde teftiş yapıyordu. Her askerin teçhizatını kontrol ediyordu. Bir eksik bulduğunda hemen tamamlattırıyordu.',
    'Yüzbaşı haritaya bakıyordu. Düşmanın yarın nereden saldıracağını tahmin etmeye çalışıyordu. Subaylarına strateji anlatıyordu.',
    'Asker arkadaşımı sırtımda taşıdım. Yaralanmıştı, kanaması durmuyordu. Yetmiş kilometre yürüdüm, sağlık çadırına vardım.',
    'Bir asker üzerinde yedi mermi vardı. Hepsi gögüsüne saplanmıştı. Yine de ayakta durup arkadaşlarına haber verdi, sonra düştü.',
    'Yiğit komutan kılıcını çekti, atına bindi. Adamlarını arkasından yürüttü, düşmana doğru atıldı. Bir tek meydan okuyuştu bu.',
    'Topçu eri mermisini namluya yerleştirdi. Hedef uzaktaydı, ama o tam isabetle vurdu. Düşmanın istihkâmı bir anda yıkıldı.',
    'Eski bir askerdi, savaşı sevmezdi. Ama gerektiğinde silahını eline alır, korkusuzca dövüşürdü. Vatanı için her şeyini verirdi.',
    'Şehit ailesine haber götüren çavuş kapıyı çaldı. Anne kapıyı açtığında her şeyi anladı. Sessizce yere çöktü, ağlamaya başladı.',
    'Mektep çocukları sıralarda otururdu. Hocaefendi tahtaya bir şey yazardı, herkes defterine geçirirdi. Sessizdi sınıf, kuş uçmazdı.',
    'Çocukluğumda her sabah erken kalkardım. Heybeyi sırtlanır, mektebe koşardım. Yolda diğer çocuklarla şakalaşırdık.',
    'Köy mektebinin önünde durdum bir an. Yıllar önce burada okumuş, ilk satırlarımı yazmıştım. Şimdi büyümüş, geri dönmüştüm.',
    'Hocaefendi çubuğunu çekti, dumanını saldı. Sonra bize hikâye anlatmaya başladı. Hepimiz pür dikkat dinliyorduk.',
    'Yaramaz çocuk sınıfta gülüştü. Hocaefendi öfkelendi, falakayı çıkardı. Çocuk korkuyla yere oturdu, ağlamaya başladı.',
    'Mektepten çıkınca çocuklar sokağa fırladı. Birbirleriyle koşmaca oynuyorlardı. Akşama kadar bitmek bilmezdi enerjileri.',
    'İlk şiirimi mektepte yazdım. Vatan üzerineydi, kafiyeleri sakattı. Ama hocaefendi övdü, başımı okşadı.',
    'Mektebin bahçesinde bir armut ağacı vardı. Yaz aylarında meyve verirdi. Biz çocuklar onun altında oturup ders çalışırdık.',
    'Mektep arkadaşımdı, en yakın dostumdu. Birlikte oyun oynar, birlikte ders çalışırdık. Yıllar sonra hâlâ özlemini hissediyorum.',
    'Sınav günü herkes telaşlıydı. Hocaefendi sınıfa girdi, kâğıtları dağıttı. Bir saat sonra herkes terlemiş, biri ağlıyordu.',
    'Mektep arkadaşlarımla son defa görüşmüştük. Hepimiz büyümüş, farklı yollara dağılmıştık. Ama o günleri unutmadık.',
    'Yaz tatili gelince çocuklar köyün dışına koşardı. Derelerde balık tutardık, ağaçlara çıkardık. Tatil bitince üzülerek mektebe dönerdik.',
    'Bir çocuk evden kaçmıştı. Hocaefendi haber alınca babasına yazdı. Çocuk sonra utanarak geri döndü, kulağı çekildi.',
    'Mektep müsameresi düzenleniyordu. Çocuklar şiirler ezberlemiş, müzik aletleri çalmayı öğrenmişlerdi. Aileler salonu doldurmuştu.',
    'Çocukluk hatıram bir kar yağışıdır. Sabah uyandığımda her yer beyaz olmuştu. Diğer çocuklarla kartopu oynamak için fırlamıştık.',
    'Yaşlı bir adam köyün kenarında oturmuştu. Bastonuna dayanmış, gelip geçenlere bakıyordu. Hayatın son demlerini yaşıyordu.',
    'Köyün kadınları çeşme başında toplanmıştı. Su doldururken hep birlikte konuşuyorlardı. Köyün haberleri orada en hızlı yayılırdı.',
    'Reşit Bey kasabanın en zengin tüccarıydı. Konağı şehir merkezindeydi. Fakirleri hiç unutmaz, her ramazan iftar sofrası kurardı.',
    'Köyün ihtiyarı sabah serinliğinde mezarlığa yürüdü. Eşinin mezarı başında durdu, sessizce dua etti. Sonra ağır adımlarla geri döndü.',
    'Yetim Hasan köyün gözdesiydi. Annesi babası savaşta ölmüştü. Bütün köy onu büyütüyordu, her evde bir misafir gibi karşılanıyordu.',
    'Hocaefendi köyün adaletiydi. İki kişi arasında bir mesele olsa, ona giderlerdi. Sözüne kimse karşı gelmezdi, herkes saygı duyardı.',
    'Köyün delikanlıları akşamüstü meydanda toplanırdı. Güreş tutarlardı, türkü söylerlerdi. Bazen kavga ederlerdi, sonra hemen barışırlardı.',
    'Yaşlı kadın torununa nasihat ediyordu. "Doğruluktan ayrılma" diyordu. "Babamdan duyduğum en güzel söz buydu."',
    'Çoban kavalını çıkardı, çalmaya başladı. Sürünün başında bir taşa oturmuştu. Ezgiler uzaklara, ovaya yayılıyordu.',
    'Köy demircisi sabah erken dükkânını açmıştı. Örs üzerinde çekiç sesleri başladı. Köylüler at nalı sipariş veriyordu.',
    'İhtiyar nineler köy evlerinin önünde oturmuştu. Ellerinde iğne iplik vardı, çorap örüyorlardı. Karşılıklı sohbet ediyorlardı.',
    'Çiftlik sahibi sabah erken atına bindi. Tarlalarını dolaşacaktı, çalışanları kontrol edecekti. Akşama kadar dolanırdı.',
    'Köy düğününde herkes mutluydu. Gelin başına altın takıldı, damat geleneksel kıyafetlerini giydi. Davul zurna gece yarısına kadar çaldı.',
    'Sokakta oynayan çocuklar ihtiyar bir adamı gördüler. Adam ağır ağır yürüyor, bastonla yolu yokluyordu. Belli ki gözleri görmüyordu.',
    'Köye gezgin bir derviş geldi. Eski püskü bir aba giymiş, elinde tesbih çekiyordu. Köylüler ona yemek verdi, başına toplandı.',
    'Vatan dedikleri toprak, üzerinde doğduğumuz, büyüdüğümüz, atalarımızın kanıyla suladığı yerdir. Onu sevmek, korumak bizim borcumuzdur.',
    'Türk milleti hiçbir zaman boyun eğmemiştir. Tarih boyunca pek çok düşmanı yenmiş, vatanını korumuştur. Bundan sonra da öyle olacaktır.',
    'Genç asker andını içerken titriyordu. Bayrak önünde durmuş, vatanına sadakat yemini ediyordu. Bu ona hayatının en kutsal günüydü.',
    'Anadolu toprakları kanla sulanmış kutsal bir vatandır. Her karış toprağında bir şehidin kemikleri yatar. Bunu hatırlamak gerekir.',
    'Vatan ne demektir bilir misiniz? Doğduğun, büyüdüğün, üzerinde nefes aldığın yerdir. Onu kaybedersen başka hiçbir şeyin kalmaz.',
    'Türk askerinin cesareti dillere destandır. Dünya tarihinde böyle bir ordu görülmemiştir. Düşmanları bile ona saygı duyar.',
    'Bayrak göklere çekildi, marş okundu. Bütün asker hazır ol durdu. Komutanın gözleri yaşardı, bir an konuşamadı.',
    'Şehit kanı ile yoğrulmuş bir bayrağa bakmak başka bir duygudur. Onu görünce yüreğin titrer, gözlerin dolar. Vatan budur işte.',
    'Genç bir kız oğluna mektup yazıyordu. Cephede savaşan oğluna. "Vatan için canını ver, döneme\'den utan" diyordu son satırlarda.',
    'Türk bayrağı her zaman dik durmuştur. Hiçbir rüzgar onu indirememiştir. Ne zaman ki indi, hemen bir kahraman ortaya çıkmış, onu tekrar göndere çekmiştir.',
    'Köy meydanında bayrak göndere çekildi. Bütün halk toplanmış, milli marşı söylüyordu. İhtiyarların gözleri yaşardı.',
    'Vatan müdafaası kutsal bir vazifedir. Hiç kimse bundan kaçamaz. Erkek olsun kadın olsun, çocuk olsun ihtiyar olsun, vazifesini yapar.',
    'Düşman geldiğinde köyde sadece kadınlar ve yaşlılar vardı. Erkekler cepheye gitmişti. Ama kadınlar da silahlanıp düşmana karşı durdu.',
    'Milli mücadele yıllarında her köy bir kale gibiydi. Halk düşmana yer vermedi. Vatanın bağımsızlığı için her şeyini feda etti.',
    'Asker olmak şereftir. Vatanına hizmet etmek, onu savunmak için silah taşımak en yüce iş. Her Türk genci bunu özler.',
]

print(f'Omer Seyfettin: {len(OMER_SEYFETTIN)} metin')

# Yazar 1: Sabahattin Ali (1907-1948)
# Psikolojik, melankoli, 1. tekil sahis, Berlin/gurbet
SABAHATTIN_ALI = [
    "Berlin'in soğuk sokaklarında saatlerce yürüdüğüm geceleri hatırlıyorum. Kar lapa lapa yağıyordu, kimseler yoktu. Ben ise içimdeki o ezici yalnızlıkla baş başaydım.",
    'Gurbet hayatı bana çok şey öğretti. Yalnız kalmayı, kendi kendime konuşmayı, susmayı öğrendim. Belki de en önemlisi, insanlara güvenmemeyi.',
    'Pansiyon odasında geceler birbirine benziyordu. Tek bir lamba yanıyordu, dışarıda yağmur düşüyordu. Ben pencerenin önünde sigara içiyor, sokaktan geçenleri seyrediyordum.',
    'Ev sahibim yaşlı bir kadındı. Bana hiç karışmazdı, ben de ona. İki yıl boyunca aynı evde oturduk, neredeyse hiç konuşmadık. Belki de en iyisi buydu.',
    'Almanca öğrenmeye çalışıyordum, ama dil giderek beni boğmaya başlıyordu. Bazen kendime soruyordum: ne arıyorum ben burada? Cevabı bulamıyordum.',
    'Kafede oturuyordum, karşımda bir kitap vardı. Ama okuyamıyordum. Gözüm sayfada gezindiği halde aklım uzaklarda, çocukluğumun köyündeydi.',
    "Bir sabah uyandığımda her yer beyazdı. Berlin'i ilk defa karda görüyordum. İçimde tarif edemediğim bir hüzün vardı, sanki bir kayıp yaşamıştım.",
    'Üniversitede tanıştığım birkaç kişi vardı. Onlarla zaman zaman dışarı çıkardık. Ama yine de hep yalnız hissederdim, anlatamam o duyguyu.',
    'Akşamları nehrin kenarında yürürdüm. Su yavaş yavaş akıyor, etrafında ışıklar yanıyordu. Ben elimde sigara, gözlerimde uzak bir bakışla saatlerce dolaşırdım.',
    'Yabancı bir şehirde yaşamak insanı değiştirir. Geriye eski sen kalmaz, başka biri olursun. Ben de oraya gittiğimde başka biriydim, döndüğümde başka.',
    'Yabancılarla birlikte yaşamak zor. Onların alışkanlıkları, gelenekleri sana hep tuhaf gelir. Sense onların gözünde bir merak nesnesisindir, anlaşılmaz bir varlık.',
    'Türk olduğumu söylediğimde insanlar şaşırırdı. Bizi tanımıyorlardı, hep yanlış bilgileri vardı. Ben de düzeltmeye çalışmaktan vazgeçtim bir süre sonra.',
    'Pansiyon merdivenlerini her gece geç saatlerde çıkardım. Sessiz olmaya çalışırdım, ama tahtalar gıcırdardı. Bir gün yaşlı kadın seslendi: "Yine geç geldin."',
    'Memleket hasreti garip bir duygu. Sevdiğin bir kadına özlemek gibi, ama daha derin. İçten içe bir sızı, hiç dinmeyen bir burukluk.',
    "Berlin'deki o yıllar hayatımın en garip dönemiydi. Yalnızdım, kimseyi tanımıyordum. Ama bu yalnızlık bana acı vermiyordu, aksine bir tür huzur veriyordu.",
    'Postaneye her gün uğrardım. Anneme yazdığım mektubun cevabını beklerdim. Bazen aylarca beklerdiğim cevap gelmezdi, içimde tarif edilmez bir kırgınlık olurdu.',
    'Bir yabancı şehirde sevdiğin biri yoksa, her şey kuru ve renksiz olur. Sokaklar, evler, insanlar, her şey gri bir tabaka altında kalır. Yaşamak değil, var olmaktır o zaman.',
    'Vatanımdan uzaktayken, ona dair her küçük şeyi hatırlardım. Annemin pişirdiği yemekler, mahalle çocuklarının sesi, sabah ezanı. Bunlar bana hep eksik olurdu.',
    'Onu ilk gördüğüm akşamı hâlâ hatırlıyorum. Eski bir kitapçının önünde durmuştu. Vitrindeki kitaplara bakıyordu, yüzünde tarif edemediğim bir hüzün vardı.',
    'Trende karşıma oturmuştu. Kitap okuyordu, ara sıra başını kaldırıp pencereden bakıyordu. Onunla konuşmak istedim, fakat cesaret edemedim. İndiğinde gitti, ben kaldım.',
    'Bir kadın tanımıştım, hiç unutamadığım. Sokakta yürürken yanından geçmiştim, dönüp baktı, hafifçe gülümsedi. O gülümseme yıllarca aklımdan çıkmadı.',
    'Onun gözleri vardı ya, o gözler. Karanlık bir kuyu gibiydi. İçine bakınca insan kendini kaybediyordu. Ben de baktım bir kere, sonra hiç kurtulamadım.',
    'Yıllar sonra eski bir tanıdığa rastladım. Kaldırımda durmuştuk, lafa daldık. Konuştukça gözlerinin nasıl yaşardığını fark ettim. "Çok değiştin" dedi sonunda.',
    'Onunla tanıştığım gece bir partideydik. Kalabalıkta gözlerimiz bir an birleşti. O an sanki dünya durdu, sadece o ve ben kaldık. Sonra kalabalık tekrar başladı.',
    'Sevdiğim kadın bana hiçbir zaman cevap vermedi. Yıllarca onu sevdim, kimseye söylemedim. Sadece içimde tuttum, taşıdım. Belki de aşk böyle bir şey.',
    'Vapurda karşımda oturuyordu. Saçları rüzgarda dalgalanıyordu. Ona bakıyordum sürekli, gözlerimi alamıyordum. Sonra indi, kayboldu, hiç bir daha görmedim.',
    'Bir kafede otururken karşımdaki adama bakıyordum. Yüzü tanıdık geliyordu, ama nereden hatırlamıyordum. Sonra anladım: yıllar önce mektep arkadaşımdı. Ne kadar değişmişti.',
    'İlk aşkımı hatırlıyorum. Ben on altı yaşındaydım, o on beş. Mahalleli bir kızdı, her sabah pencereden bana bakardı. Ne ben konuşabildim, ne o.',
    'Bana en derin acıyı veren kadın bile, aynı zamanda hayatımdaki en güzel günleri yaşatmıştı. İşin tuhafı, ben onu hâlâ özlüyorum, bütün acılara rağmen.',
    'Kahvede oturmuş, dışarıda yağmurun düşüşünü seyrediyordum. O da geldi, karşıma oturdu. Konuşmadık uzun süre. Sadece yan yana oturup yağmuru izledik.',
    'Bir karşılaşma vardır ki, insanın hayatını değiştirir. Bilirsiniz, o anı asla unutmazsınız. Benim hayatımda da öyle bir an vardı. Hâlâ taşıyorum onu içimde.',
    'Geçen gün eski bir fotoğrafımı buldum. Onun yanındaydım fotoğrafta. İkimiz de gülüyorduk, çok mutluyduk. Bakarken birden gözlerim yaşardı, kendimi tutamadım.',
    'Sokak başında durdu, bana baktı. O bakışta her şey vardı: pişmanlık, özlem, biraz da öfke. Ama hiçbir şey söylemedi. Sadece döndü, gitti.',
    'Onun mektubunu aldığım gün hayatım değişmişti. İki sayfa yazmıştı, son cümleyi okurken titreyen ellerle kâğıdı bıraktım. "Seni bir daha göremeyeceğim" diyordu.',
    'Birlikte geçirdiğimiz son akşamı düşünüyorum. Restoranda oturmuştuk, yemek yememiştik. Sadece bakışmıştık, sonra eve gitmiştik. Ertesi gün ayrılmıştık.',
    'Aşk insanın boynuna geçirilen en ağır ve en tatlı zincirdir. Kurtulmak istersin, ama kurtulamazsın. Hatta zamanla sevmeye başlarsın o zinciri.',
    'İnsanın içinde bazen öyle duygular olur ki, ne adını koyabilirsin ne tarif edebilirsin. Sadece bir ağırlık, sadece bir burukluk. Yıllarca taşırsın, kimseye anlatamazsın.',
    'Geceleri uyuyamazdım. Tavana bakıp eski günleri düşünürdüm. Annem nasıldı, babam ne yapmıştı, çocukluğumda hangi sokaklarda oynamıştım. Hepsi gözümün önünden geçerdi.',
    'Kendi kendime sorarım bazen: Ben gerçekten yaşıyor muyum, yoksa sadece var olmaktan ibaret bir hâlde miyim? Cevabını bulamam. Sigara içer, pencereden bakarım.',
    'İnsanın içinde tarif edemediği bir burukluk olur. Sebebini bilmediğin bir hüzün. Sabah uyandığında üzerinde, akşam yatağa girdiğinde üzerinde. Hep var, hiç gitmez.',
    'Aynaya baktım, kendimi tanımadım. Yüzüm değişmişti, gözlerim eski parlaklığını yitirmişti. Ne zaman bu kadar yaşlanmıştım? Kendime ait olmadığımı hissettim bir an.',
    'Bazen insan kendi kendine düşman olur. Kendine acı verir, kendini cezalandırır. Neden böyle yaparız bilmiyorum. Belki içimizdeki o karanlık tarafın bir oyunudur.',
    'Mutluluk dedikleri şey bana hep uzak gelmiştir. Hayatımda gülmeye değer pek az an yaşadım. Çoğu zaman bir sıkıntı, bir bunalım, bir karanlık.',
    'Bir insanı tanımak çok zordur. Yıllarca beraber oturursun, sonunda yine de bilmediğin bir tarafı çıkar karşına. Şaşırırsın, kırılırsın, ama yine de severim insanları.',
    'İnsanlar arasında en çok kendinden bahseden adamları tanırız aslında. Onlar herkesin önünde konuşur, gülerler. Ama içlerinde bir karanlık taşırlar, hiç kimse görmez.',
    'Hayat insana acımasız bir hocadır. Önce kalbini kırar, sonra kendini sevmeyi öğretir. Onun derslerini geç anlayanlar, hep eksik kalır.',
    'Bazen kendime soruyordum: Ben kimim, ne istiyorum, nereye gidiyorum? Cevaplar yoktu. Sadece sorular vardı, üst üste yığılan, ağırlaşan sorular.',
    'İnsanın en zor uğraşı kendiyle baş etmektir. Dışarıdaki dünya gelir geçer, ama kendinden kaçamazsın. Hep yanında olur, hep konuşur, hep eleştirir.',
    'Sevmek bana hep acı verdi. Belki sevdiğim kadınlar bana layık değildi, belki ben onlara. Ne olursa olsun, sevdiğim her insan sonunda beni terk etti.',
    'Yıllar geçtikçe insan değişiyor. Eskiden inandığım şeylere bugün inanmıyorum. Eskiden sevdiğim insanlardan bazılarını artık sevmiyorum. Kim bilir on yıl sonra nasıl olacağım.',
    'Bir karanlık var ki içimde, hep oradadır. Bazen unuturum, gülerim, hayatı yaşıyorum sanırım. Sonra birden hatırlarım, o karanlık hemen yüzeye çıkıverir.',
    'Geçmişimi hatırlamaktan korkardım. Çocukluğumdaki o ağır anılar, gençliğimdeki hayal kırıklıkları. Hepsi bir yerde saklı, ortaya çıkmaması için elimden geleni yapıyorum.',
    'İnsan bazen kendi içinde kayboluyor. Düşüncelerinin labirentinde dolaşıyor, çıkış yolunu bulamıyor. Ben de öyleydim, çoğu zaman kendi içimde kaybolurdum.',
    'Memurluk hayatı bana hiç yakışmıyordu. Her sabah aynı yere gidiyor, aynı işleri yapıyordum. İçim sıkılıyordu, kaçmak istiyordum. Fakat nereye? Hangi hayata?',
    "Daireye girince hep aynı yüzlerle karşılaşırdım. Şef Bey'in soğuk bakışı, arkadaşların boş gülüşmesi, eski masalar, sararmış kâğıtlar. Her şey aynıydı, hep aynıydı.",
    'Saat beşi beklerdim sabırsızlıkla. İş bittiğinde sokağa fırlardım, derin nefes alırdım. Sanki sekiz saat boyunca bir mağarada hapis kalmıştım.',
    'Memur arkadaşlarımla pek konuşmazdım. Onlar maaş, terfi, lokal yemekleri üzerine konuşurlardı. Bana bu konular ilgi çekmezdi, çoğunlukla susardım.',
    'Akşamları daireden çıkınca hep aynı kahveye giderdim. Köşede bir masa, bir bardak çay, biraz da pencereden dışarıya bakmak. Buydu hayatım.',
    'Maaş günleri herkes neşeli olurdu. Banka önünde sıraya girerdik. Ben bütün parayı annemin evine yollardım, kendime sadece sigara parası ayırırdım.',
    'Bir hayat ki, sabahtan akşama hep aynı işi yapıyorsun, hep aynı insanları görüyorsun. Yaşamak demek bu mu acaba? Hayır, bu sadece var olmak.',
    'Daireden ayrılmak istiyordum, ama nereye gidecektim? Başka bir iş yoktu, başka bir yol göremiyordum. Mecburen kalıyordum, içten içe eriyordum.',
    'Şef Bey bana her sabah aynı şeyleri söylerdi. Aynı emirleri verirdi, aynı kâğıtları doldurturdu. Ben içten içe gülerdim bazen, bazen de patlamak isterdim.',
    'Memur olmak insanın ruhunu öldürür. Her gün biraz daha eksilirsin, biraz daha sönersin. Sonunda bakarsın ki, kendine ait hiçbir şeyin kalmamış.',
    'Daire arkadaşlarımın çoğu yıllardır oradaydı. Ben yeniydim, ama onların geleceğini de görüyordum: aynı masa, aynı işler, aynı sıkıcı yıllar.',
    'Bir gün başımı kaldırdım, çevreme baktım. Herkes başını kâğıtlara eğmişti, kalemler işliyordu. Bir an düşündüm: Hayatımı burada bitirmek istiyor muyum? Cevap hayırdı.',
    'Yağmur yağıyordu. Sokaklar ıslak, evler hüzünlü görünüyordu. Şemsiyem yoktu, ıslanarak yürüdüm. Ne acelem vardı, ne gidecek bir yer. Sadece yürümek istiyordum.',
    'Akşam pencerenin önünde durdum. Sokakta yağmur yağıyordu, lambalar yanıyordu. Bir kedi ıslanmaktan kurtulmak için saçak altına sığınmıştı. Onun bile bir sığınağı vardı, ya benim?',
    'Sonbahar geldi, yapraklar dökülüyordu. Park bankında oturmuş, etrafa bakıyordum. Bir kadın geçti yanımdan, başı önüne eğikti. O da benim gibi düşüncelere dalmıştı.',
    'Pencerenin önünde saatlerce oturup dışarıyı seyretmeyi seviyordum. İnsanlar gelip geçiyordu, hayatlarına devam ediyorlardı. Ben ise bir köşede, onlardan büsbütün ayrı hissediyordum kendimi.',
    'Kar yağdığı günlerde sokağa çıkmazdım. Pencereden seyrederdim sadece. Karın o sessiz inişi, sokakların yavaş yavaş beyaza bürünmesi. Bana hep bir hüzün verirdi.',
    'Sahile inmiştim. Deniz dalgalıydı, rüzgar sertti. Kıyıda taşların üzerine oturdum, dakikalarca denizi seyrettim. İçimdeki o ağırlık, dalgalarla biraz hafiflemişti.',
    'Bir tren istasyonunda saatlerce bekledim. Trenler gelip gitti, insanlar koşturdu. Ben oturmuş, hepsini seyrettim. Benim gideceğim hiçbir yer yoktu, sadece bekliyordum.',
    'Park bankında uyumuşum. Uyandığımda hava kararmıştı, etrafta kimse yoktu. Üşümüşüm, eklemlerim tutulmuş. Yine de oradan ayrılmak istemiyordum, garip bir his.',
    'Mevsim sonbahardı, hava bulanıktı. Sokaklarda yapraklar uçuşuyordu. Ben de onlar gibi savruluyordum, rüzgâra kapılmış, nereye gideceğimi bilmeden.',
    'Akşam saatlerinde camlar buğulanırdı. Parmağımla harf yazardım o buğulu camlara. Sonra silinirdi, hiçbir iz kalmazdı. Tıpkı hayatım gibi.',
    'Bir yaz akşamı sahile inmiştim. Gün batıyordu, deniz pembeleşmişti. Az ötede bir balıkçı ağlarını çekiyordu. Bu manzara bana huzur vermişti, çok ender olan bir his.',
    'Çocukluğumdaki köyü hatırlıyorum. Sabahları erken kalkardık, annem bana taze ekmek dilimleri verirdi. Babam tarladan dönerdi yorgun, ama hep gülümseyerek.',
    'Annem hep yün eğirirdi geceleri. Lambanın ışığında otururdu, parmakları hızla işlerdi. Ben yanına büzülür, masallarını dinlerdim. Bu hatıra hâlâ canlı içimde.',
    'Babamla pek konuşmazdım. O da konuşmazdı zaten. Ama aramızda bir şey vardı, sözsüz bir anlaşma. Çiftliğe birlikte gittiğimizde anlardık birbirimizi.',
    'Anneme yazmak istiyordum, ama bir türlü kelimeleri bulamıyordum. Belki de söyleyecek bir şeyim kalmamıştı artık. Mektup kağıdı önümde duruyordu, ben de boş bakıyordum.',
    'Çocukluğumda evimizin bir bahçesi vardı. Annem orada çiçek yetiştirirdi. Ben kazımı yapardım, balçık çamur olurdu. Annem kızmazdı, sadece gülerdi.',
    'Babam vefat ettiği gün ben yedi yaşındaydım. Hiç hatırlamıyorum o günü, ama annemin gözlerindeki o sonsuz hüznü hatırlıyorum. Bir daha o bakıştan kurtulamadı.',
    'Anneme her zaman borçlu kalacağım. Beni o yetiştirdi, o büyüttü. Tek başına, yorgun, fakir, ama kararlı. Onun emeği olmasa bugün hiç olmazdım.',
    'Çocukluğumun en mutlu anısı, babamla balık tutmaya gittiğimiz bir gündür. Ben üç yaşındaydım, ama hâlâ hatırlıyorum. Ellerimden tutmuştu, suya baktırmıştı.',
    'Annem her sabah erken kalkardı. Önce sobayı yakar, sonra çay demlerdi. Ben uyanır, mutfağa gelirdim, o beni güler yüzle karşılardı. O sabahları çok özlüyorum.',
    'Babamın eski paltosu hâlâ duvardaki kancada asılı. Annem onu indirmedi, indirme demedi. Bazen yanına gidip kokluyorum. Hâlâ onun kokusu var üzerinde.',
    'Köyümüzde bir çeşme vardı, ben hâlâ rüyalarımda görürüm. Soğuk, berrak su akardı durmadan. Annemle birlikte oraya giderdik, su doldururduk. Hayatımın en güzel anları.',
    'Annem dua ederken yanına otururdum. Ne dediğini anlamazdım, ama bana huzur verirdi. Onun dudaklarından çıkan sözler, sanki beni de korurdu.',
    'O kitap onun için en kıymetli şeydi. Babasından kalmıştı. Sayfaları sararmış, kapağı yıpranmıştı. Ama her gece yatmadan önce mutlaka birkaç sayfa okurdu.',
    'Eski bir kitapçıya girmiştim. Tozlu rafların arasında dolaşıyordum. Birden bir kitap dikkatimi çekti, açtım, ilk sayfada bir yazı vardı: "Sevgilime, hep özlemle."',
    'Kitap okumak benim en büyük tesellimdi. Yalnız olduğum zamanlarda, anlatamadığım duygular yaşadığımda kitaplara sığınırdım. Onlar bana hep yoldaş oldular.',
    'Bir şiir okuyordum, gözlerim yaşardı. Şair, benim hissettiğim ama söyleyemediğim duyguları yazmıştı. Bu kâğıt parçasının böyle bir gücü olmasına şaşıyordum.',
    'Yazmaya başlamıştım o günlerde. Geceleri lamba ışığında oturur, sayfaları doldururdum. Sonra sabah okuyup yırtardım çoğunu. Yazdıklarım hiçbir zaman yeterli gelmezdi bana.',
    'Bir kitabevinde tanıştık. İkimiz de aynı kitaba uzanmıştık. O gülümsedi, ben de gülümsedim. Sonra bir kahveye gittik, konuştuk. O akşamdan sonra hayatım değişti.',
    'Bazı kitaplar vardır, insanın hayatını değiştirir. Bana da öyle bir kitap rast geldi yirmi yaşımda. Bitirdiğimde başka biri olmuştum, eski ben yoktu artık.',
    'Sevdiğim şairin bir mısrası vardı, hep tekrarlardım. "Yalnızlık bir kuyu gibidir" derdi. Yıllar sonra anladım ne demek istediğini, o kuyuya kendim de düştüm çünkü.',
    'Kütüphanede oturmuş, eski bir kitabı okuyordum. Karşımdaki masada bir kadın da kitap okuyordu. Ara sıra bakıyordum ona, o da ara sıra bana. Ama hiç konuşmadık.',
    'Bir defter aldım, içine her gün düşüncelerimi yazıyordum. Kimse okumayacaktı, sadece kendim için. Yıllar sonra okuyunca o kadar çok unutmuşum ki şaşırdım.',
    'Sigara içmeyi gençken öğrendim. Önce nefes alamıyordum, sonra alışırım. Şimdi günde bir paket bitiriyorum. Belki kötü, ama bana eşlik eden tek dost o.',
    'Gecenin sessizliğinde sigara içerdim. Pencere kenarında otururdum, dumanı havaya bırakırdım. O dumanın yavaş yavaş kaybolması bana garip bir huzur verirdi.',
    'Geceleri hep uyanık kalırdım. Şehir uyurken ben uyanıktım, düşünürdüm. Sabaha karşı zorlukla uyurdum, öğleye kadar uyumak isterdim, olmazdı.',
    'Sigara dumanı arasından dışarıyı seyrederdim. Sokaktan geçen insanların hepsinin bir hayatı vardı, bir hikâyesi vardı. Ben ise sadece izleyendim, kendi hikâyemi unutmuş gibi.',
    'Sessizlik bazen çok ağır olur. Yalnız bir odada, gecenin bir vaktinde sessizliği dinlemek korkutucu olabilir. Ben alıştım, ama yine de bazen ürperirim.',
    'Sigaramın ucu söndü, fark etmedim. Düşünceler içinde kaybolmuştum. Kendime gelince elimdeki sönmüş izmariti gördüm, hafifçe gülümsedim, başka bir tane yaktım.',
    'Saatlerce konuşmadan oturabilirdim. Yalnızlık beni rahatsız etmiyordu artık. Hatta zamanla aramızda bir tür dostluk kurulmuştu, ona alışmıştım.',
    'Hayat bana her zaman gece gibi geldi. Karanlık, uzun, sessiz. Bazen yıldızlar görünür, bazen ay çıkar. Ama gündüz, gerçek aydınlık, hiç gelmedi.',
    'İnsan bazen kendi hayatına yabancılaşıyor. Yaptıklarına, söylediklerine, hatta sevdiklerine. Sanki başka biri yaşıyor, sen sadece izliyorsun. Bu hisse alışmak zor.',
    'Hayatın bir anlamı var mı bilmiyorum. Olduğunu söyleyenler vardı, olmadığını söyleyenler de. Ben kendi anlamımı bulamadım, belki de aramayı yanlış yerde yapıyordum.',
    'Bazen rüyamda eski evimi görürüm. Annem mutfakta yemek pişiriyor, babam koltukta oturuyor. Sonra uyanırım, gerçeği hatırlarım, gözlerim yaşarır.',
    'İnsan bazen hayatında öyle anlar yaşar ki, sonradan bütün hayat onun etrafında dönmeye başlar. Benim için de öyle olmuştu. Onu tanıdığım gece, dünya değişmişti.',
    'Geçmişe takılı kalmaktan kurtulamıyordum. Yıllar geçti, ama eski hatıralar hep tazeydi. İlk aşkım, ilk hayal kırıklığım, ilk yalnızlık. Hepsi hâlâ canlı içimde.',
    'Gelecek hakkında düşünmek istemiyordum. Bilinmeyen beni korkutuyordu. Belki yarın ölürdüm, belki on yıl yaşardım. Hangisi olursa olsun, ben sadece bugünü yaşamaya çalışıyordum.',
    'Yaşamak değil de, hayatta kalmaktı yaptığım. Her gün uyanıyor, yemek yiyor, sokağa çıkıyor, yatağa giriyordum. Ama gerçekten yaşadığımı hissetmiyordum. Sadece nefes alıyordum.',
    'Bir gün ölüm hakkında uzun uzun düşündüm. Korku duymadım, sadece bir merak. Ne olacaktı acaba? Bilmiyorum, ama bir gün öğreneceğim, herkes gibi.',
    'Hatıralarımı bir kutuya koymak isterdim. Hepsini içine atıp denize bıraksaydım keşke. Belki dalgalar onları başka kıyılara taşırdı, ben de kurtulurdum.',
    'İnsan zamanla değişiyor, ama yine de bir özü kalıyor sanırım. Ben kim olursam olayım, derinlerimde hep aynı çocuk var. O on yaşındaki, yalnız, içe kapanık çocuk.',
    'Onunla son görüşmemiz bir vapurda olmuştu. İki saat boyunca yan yana oturmuştuk. Hemen hemen hiç konuşmadık. İndiğimizde el sıkıştık, başka bir şey demedik.',
    'Sevdiğim insanları hep kaybettim. Önce babam öldü, sonra dayım, sonra arkadaşlarım birer birer uzaklaştı. Şimdi yalnızım, kim bilir ne kadar daha böyle olacak.',
    'Bir mezarlığa giderdim her ayın ilk Cuması. Babamın mezarı oradaydı. Saatlerce başında oturur, ona içimden konuşurdum. Cevap alamazdım, ama yine de konuşurdum.',
    'Veda etmek hiç kolay olmadı bana. Sevdiğim birinden ayrıldığımda hep ağlardım. Yıllar geçtikçe ağlamayı bıraktım, içimde sıkışıp kaldı her şey.',
    'Birini özlemek garip bir duygu. Yanında değildir, ama her yerde onu hissedersin. Sokakta yürürken, yemek yerken, uyumadan önce. Hep o vardır, hep yanındadır görünmez bir biçimde.',
    'Geçen yıl yaşlı bir adamı uğurladık. Hayatımdaki son baba figürüydü. Cenazede ağlayamadım, kuru bir hüzün vardı içimde. Sonradan, evime gidince, sessizce ağladım.',
    'Onun kokusunu hâlâ hatırlıyorum. Bir parfüm değildi, kendi kokusuydu. Onu öptüğümde yanağında o kokuyu duyardım. Yıllar geçti, hâlâ hatırlıyorum o kokuyu.',
    'Annemin ölümünden sonra her şey değişti. Ev başka bir evdi, sokaklar başka sokaklardı. Sanki o gittiğinde bütün dünya rengini değiştirmişti, hepsi soluk olmuştu.',
    'Akşam yine kafeye gittim. Hep aynı masada oturuyordum, en arka köşede. Garson beni tanıyordu, sipariş bile sormadan çay getirirdi.',
    'Kahvede oturuyordum, gözüm dışarıdaki yağmurdaydı. Camlar buğulanmıştı. İçerisi sıcaktı, ben üşümüştüm. Bir bardak çay daha istedim garsondan.',
    'Eski bir kahveydi orası. Duvarlarda eski fotoğraflar, tavanda yıllarca biriken sigara dumanı izi. Yaşlı adamlar tavla oynardı orada, ben de bir köşede otururdum.',
    'Bir akşam kafede otururken bir adam yanıma geldi. "Yer yok" dedi, oturdu. İki saat konuştuk, hayat hikâyesini anlattı. Sonra kalktı, gitti, hiç bir daha görmedim onu.',
    'Çayımı yudumlarken karşı masaya bakıyordum. İki kadın oturmuştu, bir şeyler anlatıyorlardı. Sesleri duyulmuyordu, ama yüz ifadelerinden mutlu olduklarını anladım.',
]

print(f'Sabahattin Ali: {len(SABAHATTIN_ALI)} metin')

# Yazar 2: Huseyin Rahmi Gurpinar (1864-1944)
# Mahalle hicvi, diyalog, ironi, dedikodu
HUSEYIN_RAHMI = [
    'Mahallenin meşhur dedikoducusu Naciye Hanım, soluk soluğa kapıdan içeri daldı. "Ay komşucuğum, duydun mu olanları?" diye haykırdı. Pakize Hanım merakla yerinden fırladı, hemen kahveyi pişirmeye koştu.',
    "Komşu Ümmü Hanım'ın sözleri her zaman dolambaçlıydı. Ne demek istediğini anlamak için zekânın yetmesi gerekirdi. Kocası bile elli yıllık evlilikten sonra hâlâ sözlerini çözemiyordu.",
    'Mahalledeki üç kadın çeşme başında toplanmıştı. Her biri elindeki bakracı doldurmadan bir araya gelirler, akşama kadar mahallenin haberlerini tartışırlardı. Çeşmenin başında dünya kurulurdu.',
    'Cuma günü mahallenin hamamı kadınlarla dolup taşıyordu. Naciye Hanım orada da dilini durduramıyordu. Yanındaki tellak, anlatılan dedikoduları işiterek başını sallıyordu.',
    'Pencereden komşu kapısına bakan Pakize Hanım, gelen gideni sayıyordu. Akşam kocası geldiğinde rapor edecekti. Hangi kadın hangi saatte gelmiş, ne kadar oturmuştu.',
    'Yengem her gün öğleden sonra komşuya gider, akşama kadar dedikodu yapardı. Eve döndüğünde mahallenin bütün haberlerini bilirdi. Amcam onu dinlemekten bezmişti, ama susmaya da cesaret edemiyordu.',
    'Mahallenin terzi dükkânı kadınların buluşma yeriydi. Bir tek elbise dikilmek için saatlerce kalır, oradan oraya laf taşırlardı. Terzi de zaten bu yüzden bir türlü iş bitiremezdi.',
    'Hatice Hanım her sabah pencereden bakardı. Mahalledeki bütün kadınların ne giydiğini ezbere bilirdi. Akşam kocasına "Filanca hanım bugün şu rengi giymiş" diye anlatırdı.',
    'Sabaha karşı bir bağırış kopmuş. Pencerelere koşan kadınlar, sokağın ortasında iki erkeği yakalamış. Bir saat sonra mahalleli toplanmış, dünyanın hâli üzerine konuşuyordu.',
    'Pakize Hanım\'ın gelinine takıldığı şeyler arasında en ilginci kahvesinin kıvamıydı. "Köpüğü olmayan kahve, kahve değildir" derdi her sabah. Gelin sabırla dinler, ama hiç kıvamı tutturamazdı.',
    'Komşunun kapısından ne çıkıyor ne giriyor, hepsini biliyordu Pakize Hanım. Sabah erkenden pencereye geçer, perdenin arkasından gözetlerdi. Mahalleli ona güler ama ondan da çekinirdi.',
    "Ay komşum, vallahi billahi şaşılacak şeyler oldu sabahleyin. Filan hanımın oğlu, bizim Hatice'nin kızına talip olmuş! Sen söyle, dünya nereye gidiyor?",
    'Kadınlar bir araya gelince muhabbet hiç bitmezdi. Önce çocuklardan başlar, sonra kocalara geçer, sonra fiyatlardan dem vurur, en son mahalle haberlerine dönerlerdi. Saatler nasıl geçtiğini anlamazlardı.',
    "Naciye Hanım'ın özelliği, hiçbir dedikoduyu kendine saklayamamasıydı. Duyduğunu hemen başka birine anlatması gerekirdi. Yoksa içinde bir yangın olur, rahat edemezdi.",
    'Mahallenin ortasında oturan Reşadiye Hanım, üst kat penceresinden her şeyi görürdü. Hangi adamın ne zaman eve geldiğini, hangi hanımın hangi gün hamama gittiğini ezbere bilirdi.',
    'Cumartesi günü ev sahibi pirinç pilavı yapacaktı. Bütün mahalleli kadınlar erken erken geldi. Sözüm ona pilav yapmaya yardıma, aslında dedikoduya. Pilav nihayetinde unutuldu, üç saat sonra hatırlandı.',
    'Komşu kapısı her vurulduğunda Pakize Hanım koşa koşa pencereye giderdi. Kim gelmiş, ne istemiş, niçin gelmiş, hepsini öğrenmeliydi. Yoksa o gece uyuyamazdı.',
    'Sokağa çıkan iki kadın eski bir tanışlıkları olduğunu hatırladı. Hemen orada laflamaya başladılar. Birinin elinde paket vardı, sokağın ortasında bıraktı. İki saat sonra yerden alıp eve döndü.',
    'Bizim mahallenin imamı, her cuma vaazında ahalinin ahlaksızlığından dem vurur. Ama kendisi de evine giderken pastacının vitrinini seyretmekten gözünü ayıramaz. İnsanlık böyle bir şey işte.',
    'Komşumuz Şükrü Efendi, hanımına müthiş düşkündü. Ne dese yapardı, ne istese alırdı. Mahallede herkes "Ne kadar iyi koca" derdi. Halbuki adamcağız evde söz hakkı bulamadığından böyle yapıyordu.',
    'Hatice Hanım, kızına koca aramaya başladı. Ama beğenisi öyle yüksekti ki, gelen taliplerden hiçbiri makbule geçmedi. "Bizim kız sıradan adama varmaz" diyordu. Kız evde kaldıkça kaldı.',
    'Mahallenin en zengin adamı Hacı Ahmet Efendi cimriliğiyle meşhurdu. Çocuğunun düğününde bile bir tabak baklava ısmarlamaya gönlü razı olmadı. "Para harcamak günahtır" derdi her vesileyle.',
    'Beyefendi sokaktan geçen güzel bir kadına bakmamak için kendini zor tuttu. Halbuki yanında hanımı vardı. Hanım fark etti, suratını astı. Akşam yemeğinde tabaklar adam aleyhinde uçuştu.',
    'Anneanne torununa boyuna nasihat verirdi. "Kızım, evlenince kaynanana hizmet et, kocana karşı gelme." Halbuki kendisi gençken kayınvalidesini günlerce sürtmüştü. İnsan unutuyor.',
    'Mahalledeki cinli evin önünden geçerken herkes besmele çekerdi. Sadece akşamcı Cemal Bey, o evin önünden sallanarak geçer, hiçbir şeye aldırmazdı. "Cinler benden kaçar" derdi gülerek.',
    'Kayınpederim öyle bir adamdı ki, ne dese öbürü tersini yapardı. Damatlığa ilk geldiğim yıl bunu fark ettim. Ondan sonra ne istesem, tersini söyledim, hep istediğim oldu.',
    'Mahallenin pintisinin biri sabaha kadar pencereden dışarıyı gözetlerdi. Bir kıvılcım görmüş, hemen aşağı inmişti. Anlaşılan komşu çocukları sokakta sigara içiyordu. Adam memnun, ihbar etmek için karakola koştu.',
    'Naciye Hanım gelinine takılırdı boyuna. Halbuki kendi de zamanında nasıl gelindi onu unutmuştu. "Benim zamanımda gelinler dik durur, başını eğerdi" diyordu, halbuki kayınvalidesine kafa tutmuştu.',
    'Müderris efendi ahaliye din dersi verirdi. Ama kendisi de gizliden gizliye eğlenirdi. Bir gece meyhaneden çıkarken görüldü, ertesi gün yine vaaz veriyordu. Kimsenin haberi yokmuş gibi yaptı.',
    'Eski dostlardan biri ortaya çıkmıştı, bir iş için para istiyordu. Hacı Ahmet hemen meslek verdi: "Sen de benim gibi cimri ol, paran birikir." Adam başını salladı, geri döndü.',
    'Mahallemizin softası vardı, her şeye karşı çıkardı. Çocuklar oyun oynasa günah, kadınlar gülüşse günah. Halbuki kendisi gece geç saatte sokakta gezerdi, kimse sormazdı niye?',
    'Şehir yenileniyordu, eski mahalleler yıkılıyordu. Hatice Hanım dert yandı: "Ah, bizim eski evimiz, kaybolacak." Halbuki yıllardır kira ödüyorlardı, ev bile kendilerinin değildi.',
    'Mahallenin kadıncağızı dul kalmıştı. Komşular önce çok üzüldü, sonra dedikoduya başladı. "Kocası ölmeden gözünü dikti komşumuza" diye konuştular. Garibanın bir lokma ekmeği bile bin lafa neden oldu.',
    'Bir adam paraya kavuştuğunda büsbütün değişir. Eskiden alçakgönüllü olan, şimdi kibirlenir. Komşulara selam vermez, eski dostları unutur. İnsan hâli işte, parayla birlikte değişiyor karakter.',
    'Mahalleliyi en çok ilgilendiren konu hep komşunun cebindeki para. "Ne kadar maaş alıyor?" "Ev ne kadara çıktı?" "Tavuğun fiyatı kaç?" Bunlar bitmek bilmezdi.',
    'Akrabalar arasında en çok kim daha zengin tartışılırdı. Bir miras işi çıktığında ilk önce avukatlar geliyordu. Akrabalık sözleri bitiyor, mahkeme başlıyordu. Para her şeyi bozardı.',
    "Reşat Bey'in en büyük korkusu kedilerdi. Sokakta bir kedi görse hemen yolu değiştirir, geriye dönerdi. Mahalleli bunu fark etmiş, çocuklar oyun olsun diye kedileri ona sürerdi.",
    'Akşamcı Cemal Bey her gece meyhaneden döner, sokakta düşe kalka eve giderdi. Karısı kapıda bekler, onu içeri sokmamak için elinden geleni yapardı. Adamcağız kapıya yaslanarak uyurdu çoğu zaman.',
    'Kahvede oturmuş gazete okuyan Behçet Bey, birden kahkahayı bastı. Yanındakiler merakla "Ne oldu, neye gülüyorsun?" diye sordular. Adam göz yaşları içinde "Hiçbir şey" diyebildi sadece.',
    'Mahalle bekçisi gece yarısı sarhoş bir adamı tutmuştu. Adam ayakta duramıyor, durmadan saçma sapan konuşuyordu. Bekçi onu karakola götürürken "Ben paşayım, bana karışamazsın!" diye bağırıyordu.',
    'Hasip Efendi yıllardır aynı paltoyu giyerdi. Hem rengi solmuş hem astarı yırtılmıştı. Hanımı "Yenisini al" derdi her bahar. O da "Bu daha gayet iyi" diyerek konuyu kapatırdı.',
    'Bizim mahalleli Hafız Mustafa, her sabah camide ezan okurdu. Sesi tatlıydı, ama nedense ezan okurken ara sıra öksürürdü. Mahalleli gülerdi, o kızmazdı, gülerek devam ederdi.',
    'Mahallenin softa adamı her cuma camiye en önden otururdu. Ama nedense her cuma uyuklardı. Hocaefendi vaaz verirken kafası düşer, sonra irkilerek kalkardı. Yan komşusu dürterdi onu.',
    'Şişman Naime Hanım merdivenleri çıkarken nefes nefese kalırdı. Üst kat komşusunun kapısına vardığında konuşamazdı. Komşu onu içeri buyur eder, hemen su getirir, otururdu Naime Hanım kanepeye.',
    "Mahalleye yeni gelen Murat Efendi'nin garip bir alışkanlığı vardı. Sokakta yürürken kendi kendine konuşurdu. İnsanlar gülerdi ona, o aldırmazdı, devam ederdi konuşmaya.",
    "Bizim mahalleli Süleyman Efendi'nin marifet yoktur ki olmasın. Demir döğer, ayakkabı tamir eder, hatta diş çekerdi. Mahalleli ona güvenir, her işine onu çağırırdı. Bedavaya yapardı çoğu zaman.",
    'Kahvedeki tavla ustası Hasan Bey hiçbir oyunu kaybetmezdi. Hile mi yapardı, marifet mi vardı bilinmez. Kim ona oyun açsa parasını veriyordu. Sonra herkes ona oyun açmaktan vazgeçti.',
    'Pazardaki pazarcı Veli, malını alanlara sürekli laf atardı. "Bu domateslerin tadına bakacaksın mübarek, balaban gibi!" diye bağırırdı. Mahalleli hem domates alır hem laflarına gülerdi.',
    'Bir doktor vardı mahallede, herkesin hastalığını okşamadan tedavi ederdi. Hasta ne derse desin "Sen merak etme, ben anladım hastalığını" derdi. Çoğu zaman yanıldığı olurdu, ama özgüvenle çalışıyordu.',
    'Komşumuz Lemİye Hanım her sözüne "Mübarek" diye başlardı. "Mübarek bugün hava güzel", "Mübarek bu ekmek taze", "Mübarek torunum çok yaramaz." Kimse bu alışkanlığı çıkaramamıştı ondan.',
    'Demirci Ahmet Usta her sabah dükkânı açarken yüksek sesle türkü söylerdi. Mahalleli o ses gelmeyince "Aman, ne oldu Ahmet Usta\'ya?" diye merak ederdi. Hasta olduğunu anlardı herkes.',
    'Kayınvalidemle aramızdaki ilişki ilk günden bozuktu. Bana her şeyi yanlış yapıyorum derdi. Yemekler tuzlu, çamaşırlar kirli, ev tertipsiz. Hiçbir şey beğenmezdi, doğrusunu kendi yapardı.',
    'Yeni gelin gelmişti eve. Kayınvalidesi onu denemek için bin türlü iş veriyordu. Gelinin sabrı taşıyordu yavaş yavaş. Bir gün patlayacaktı, herkes bunu hissediyordu.',
    'Damatlık kolay iş değildir, en zor mesleklerden biridir. Kayınpederin sözüne, kayınvalidenin yüzüne, baldızın naklerine katlanacaksın. Bütün bunlar bir kadına sahip olmanın bedelidir.',
    'Bizim evde annem hâkimdi, babam sadece yağ getirip götürürdü. Ama herkes babamı patron sanıyordu. Aslında bütün kararları annem alıyordu, sonra babamın ağzından duyuruyordu.',
    'Kız kardeşim damatlık yapmaya başladı eniştem ile. Yıllardır küçük kavgalar büyük kavgalara dönüştü. Şimdi ayrıldılar. Doğal olarak akrabalar ikiye bölündü, herkes bir tarafı tuttu.',
    'Anneannem her cumartesi torunlarına yemek yapardı. Bütün aile toplanırdı sofrada. O sofralar olmasa belki birbirimizi yılda bir görürdük. Her şeyimizi ona borçluyduk biz torunlar.',
    'Babamla annem her şey için tartışırlardı. Hangi kanalı açacaklar, ne yiyecekler, kime gidecekler. Hiçbir konuda anlaşamazlardı, ama sonunda annem hep kazanırdı. Babam mırıldanarak razı olurdu.',
    'Halamla annem yıllardır küs. Sebebini ben hâlâ tam anlamış değilim. Bir bayramda bir laf gitmiş, bir cevap dönmüş, ondan sonra hiç görüşmüyorlar. Akrabalık böyle bir şey işte.',
    'Düğün hazırlıkları başlamıştı, mahalleli iki gruba bölündü. Bir grup damadın aile dostlarıydı, diğeri gelinin. Hangi grup düğüne kadar daha çok ev gezerse, o tarafın itibarı yüksek sayılacaktı.',
    'Çocuklar küçükken anneleri çok çekerdi. Bir tanesi düşer, biri kavga eder, biri ağlardı. Akşam babası eve geldiğinde "Çocuklar nasıl?" derdi. Anne "İyiler" diye cevap verirdi her seferinde.',
    'Anneler kızlarını kıskanır derler ya, doğrudur. Benim annem de evlendiğimde mutlu görünüyordu, ama içinden üzülüyordu. Sonra anladım, çocuklarımız büyüyünce belki ben de aynı duyguyu yaşayacağım.',
    'Evin reisi babamdı sözde, ama gerçek reis evimizdeki kediydi. Onun istediği saatte yenir, onun istediği yerde otururdu. Babam onun karşısında çaresizdi, hatta korkardı sanki.',
    'Mahalledeki perili evin önünden geçerken herkes hızlanırdı. Hatta bazıları çocuklarını koştururdu. Akşamcı Cemal Bey bile bir gece geç saatte oradan geçti, derler ki saçları o gün ağarmış.',
    'Halamın eline yıllarca kına yakılmazdı. "Cinler kına kokusuna gelirler" derdi. Hâlâ inanır mı bilmem, ama torunlarına da öğretti bu hikâyeyi. Şimdi onlar da çekiniyor.',
    'Falcı kadın mahalleye geldiğinde bütün hanımlar etrafına toplanırdı. Sırayla ona fincan açtırırlardı. Çıkan tabloları akşam kocalarına anlatırlar, kocalar başlarını çeviridi sıkıntıyla.',
    'Anneannem her sabah erkenden kalkar, eve nazar değmesin diye tuz yakardı. Sokağa çıkarken üzerimize üfler, mavi boncuk takardı. "Kötü göz değmesin" diye dua ederdi.',
    'Sabaha karşı sokaktan geçen siyah kedi mahalleli için kötü işaretti. "Bu gün bir uğursuzluk gelecek" denirdi. Akşam tencereler yere düşse, bunu sabahki kediye bağlarlardı.',
    'Anneannem rüya tabir eder, fal bakar, türlü şeyler yapardı. Komşular saatlerce kapıda beklerdi. Halbuki kendisi bile en sonunda inanmazdı söylediklerine, ama söylerdi. Para da almazdı, bedavaya.',
    'Mahalleye gelen Çingene kadın el falı bakardı. Hanımlar gizlice çağırırlardı evine. "Yakında zengin olacaksın, beklenmedik para gelecek" derdi. Hanımlar inanır, on yıl beklerdi parayı.',
    'Müslüm dede hâlâ cinlerden bahsederdi. Çocukken gördüğünü anlatırdı sürekli. Hiç şüphesi yoktu, bütün cinleri ezberlemişti. Mahallenin çocukları onu dinlemekten korkardı.',
    'Komşu kadının çocuğu hastalanmıştı. Doktoru çağırmadı, hemen hocaya gitti. Hocaefendi muska yazdı, çocuğun boynuna takıldı. Çocuk iyileşti tesadüfen, kadın hocadan emindi artık.',
    'Mahallede bir hoca vardı, herkes ona koşardı. Aşk muskası, evlilik muskası, zengin olma muskası. Hocaefendi hepsini yazardı, her birini ayrı fiyata satardı. Mahalleli mutluydu, hoca da.',
    'Çocuğa nazar değdiğine inanıldığında kurşun döküldü. Anneannem yapardı bunu en iyi. Tencerede su, üzerine bir tepsi, içine kurşun. Yapılan şekillerden çocuğa kim baktığını anlardı.',
    'Bir cenaze evinde ölünün ruhunun kırk gün dolaştığına inanılırdı. Akşamları perdeleri kapatırlar, ışıkları söndürürler, sessizce otururlardı. Ölünün döneceği akrabaları rahatsız etmemesini istiyorlardı.',
    'Cuma günleri ölmek hayırlı sayılırdı. "Şu mübarek, Cuma günü gözünü kapattı" denirdi. Ailesi ondan dolayı teselli bulurdu. Halbuki ölen ölmüş, hangi gün olduğu artık fark etmezdi.',
    'Mahallenin meydanında iki çocuk kavga ediyordu. Birinin annesi pencereden onu görmüş, hemen aşağı inmişti. Diğer çocuğa söylenmeye başladı. Sonra o çocuğun annesi de inmişti, kavgayı kadınlar tamamladı.',
    'Sokağın sonundaki köşede bir gazete bayisi vardı. Sabah erkenden açılır, bütün mahallelinin geldiği yerdi. Gazetenin başlığını okuyup yorum yapanlar, gazete almadan ayrılırlardı.',
    'Pazar yerinde iki kadın kavga ediyordu. Sebep, bir demet maydanozdu. Etraflarında halka olan ahali, eğlene eğlene seyrediyordu. Sonunda birinin başörtüsü düştü, herkes kahkahayla güldü.',
    'Mahalleye bir kumpanya geldi sokakta gösteri yapacak. Çocuklar etrafa toplanmıştı. Bir adam ip üzerinde yürüdü, bir başkası ateş yuttu. Sonra şapka uzattılar, ahali metelik atmadı, kumpanya çekilip gitti.',
    'Bir sabah uyandık ki, mahalleye yeni komşu taşınmış. Akşama kadar herkes geldi geçti onların kapısından. Bir hoş geldin demek bahanesiyle, asıl niyet yeni komşuyu incelemek, sonra dedikodusunu yapmaktı.',
    'Ramazan ayında mahalle başka bir hale girerdi. Sokakta dolaşanlar bile yavaş yürürdü. İftara yakın bütün evlerden lezzetli kokular gelirdi. Top patlayınca bütün mahalle sofralara dökülürdü.',
    'Yağmurlu bir günde mahalle sular altında kalırdı. Drenaj yoktu çünkü, herkes evlerinde mahsur kalırdı. Çocuklar bunu fırsat bilir, evlerden çıkıp sularda oynarlardı. Anneleri kızar, ama yapacak bir şey yoktu.',
    'Bir gün mahalleye polis geldi. Komşu evden bir gürültü gelmiş, biri ihbar etmiş. Polisler kapıyı çaldı, içeriden ses gelmedi. Bir saat sonra anlaşıldı ki, kimse yokmuş içeride. Komşular yine merakla kapılarda toplanmıştı.',
    'Bayram sabahları mahalle çocuklarla dolup taşardı. Hepsi yeni elbiseler giymiş, ellerinde harçlık kutuları. Kapı kapı dolaşır, büyüklerin elini öperdi. Cebleri akşama kadar şişerdi parayla.',
    'Mahalleli her cuma akşamı meydanda toplanırdı. Çocuklar oyun oynar, kadınlar sohbet eder, erkekler tavla oynardı. Bu gelenek yıllarca sürdü, sonra şehir değişti, herkes evlerine kapandı.',
    'Yangın çıkmıştı mahallede, herkes sokağa fırlamıştı. Bekçi tulumbacıları çağırmak için koşmuş, gelene kadar mahalleli kovalarla suyu götürdü. Sonunda yangın söndü, ama bir ev kül oldu.',
    'Sokakta bir adamı dövüyorlardı. Mahalleli pencerelerden bakıyordu, kimse müdahale etmiyordu. Sonradan anlaşıldı ki, dövülen adam mahallenin hırsızıydı. Polis geldiğinde dövenler kaçmıştı, adam yerde kalmıştı.',
    'Mahallenin imamı her cuma vaazında ahlaktan dem vurur, halkın günahlarını sayıp dökerdi. Halbuki kendisi de bizim mahalleden biriydi, eksiklerini hepimiz bilirdik. Yine de saygıyla dinler, dışarı çıkar, ardından gülerdik.',
    'Hocaefendi mektepte çocuklara ahlak dersi verirdi. Ama kendisi mahallenin manavıyla kavga ederdi her sabah. "Domateslerin fiyatı yüksek" diye gürültü çıkarırdı. Çocuklar bunu görür, ama hocaefendiye söylemezdi.',
    'Komşu Hacı Efendi yıllardır mahalleye hac hikâyeleri anlatırdı. Halbuki bir kez gitmişti, otuz sene önce. Şimdi sanki her yıl gidiyormuş gibi konuşuyordu. Mahalleli onu nazikçe dinler, ama içinden gülerdi.',
    'Kapı önünde sadaka dilenen yaşlı bir adam vardı. Mahalleli ona göz ucuyla bakar, çoğu zaman geçerdi. Halbuki Cuma vaazlarında sadaka verme nimetinden bahsedilirdi. Söz başka, iş başkaydı.',
    'Hocaefendi cuma vaazında dünya malına meyledilmemesi gerektiğini söylerdi. Halbuki kendisi de mahallenin en zengin adamlarından biriydi. Üç ev, beş dükkân sahibiydi. Vaazlar bittiğinde rahatça evine giderdi.',
    'Bizim mahallenin softası bütün dinleri eleştirirdi. Halbuki kendi de mahallede en başında oturur, en gür sesle âmin derdi. İkiyüzlülük insanın damarına işlemiş, kolay kolay çıkmaz.',
    'Mahalleli birinin ölümünden sonra hep güzel sözler söylenir. "İyi adamdı, dürüst adamdı." Halbuki yaşarken sürekli dedikodusu yapılırdı. Ölünce melekleşirler insanlar, hep böyledir.',
    'Akşamcı Cemal Bey her gün meyhaneye giderdi, ama Cuma günü camide en ön safta dururdu. Hocaefendi de bilirdi onu, ama hiç şikâyet etmezdi. "Allah yarlığasın" derdi, kapatırdı meseleyi.',
    'Yeni eve taşınan kibar Mahir Bey, mahalleyi gözden geçirmeye gelmişti. Şıkıyak giyinmiş, bıyıklarını burmuştu. Komşulara selam verirken yarı boy eğdi, kibarlığı belliydi. Mahalleli onu ilk gördüğünde "Bu adam bizden değil" dedi.',
    'Madam Lemise gibi şıklığa düşkün bir kadın daha görmemiştim. Her gün başka bir elbise, başka bir şapka. Üst kat penceresinden bakanlar, akşama kadar onun elbiselerini sayardı.',
    'Mehmet Beyefendi her sabah saat dokuzda dışarı çıkar, akşam dokuzda dönerdi. Mahalleli ona bakıp saatlerini ayarlardı. Bir gün on dakika geç kaldı, mahalleli telaşa düştü.',
    "Yeni komşu Hayriye Hanım'ın salonu, mahallenin en şık salonuydu. Konak hanımlığından gelmişti. Komşuları toplayıp çay vermişti, ama kimse ona yetişemiyordu. Üç gün konuşuldu mahallede o davet.",
    "Mahalledeki en kibar kişi Burhan Beyefendi'ydi. Hiç kavga etmez, kimseye kötü söz etmezdi. Halbuki çocukluğunda ne haşarıydı. Yaşı ilerleyince bu kadar değişti, kimse inanamıyordu.",
    "Pakize Hanım'ın gelini şehirden gelmişti. Mahalleye uymakta zorlanıyordu. Komşularla nasıl konuşulur, çeşme başında ne yapılır, hamamda nasıl davranılır bilmiyordu. Kayınvalidesi ona her şeyi öğretmeye çalışıyordu.",
    'Tahsil görmüş Necati Bey, mahallenin entelektüeliydi. Sürekli kitap okur, gazete takip ederdi. Komşularla pek konuşmazdı, onların seviyesinden uzaktı. Halbuki kendisi de mahalleliydi, ama unutmuş gibiydi.',
    'Eski mahalleli Şükriye Hanım hayatında hiç şikâyet etmemişti. Kocası vefat etmiş, dört çocuğunu tek başına büyütmüştü. Her seferinde kendi yapmıştı, kimseye yük olmamıştı. Mahalleli onu çok severdi.',
    "Kibar Cevdet Bey'in evine her akşam misafirler gelirdi. Şehirden, mahalleden, akrabalardan. Hanımı çay servisi yapar, beyefendi sohbet ederdi. Mahalleli onların yaşamına imrenirdi.",
    "Sokak başındaki Hatice Hanım'ın evi de mahallede ünlüydü. Hayır işleriyle uğraşırdı. Fakirlere yardım eder, yetimleri okuturdu. Komşuları onu överdi, ama bazıları içten içe çekemezdi, gönülleri kara.",
    'Mahalleye yeni gelin gelmişti, çok sessiz bir kadındı. Kayınvalidesi onunla aydı baş edemiyordu. "Bu kız nasıl gelin oldu?" diye soruyordu komşularına. Komşular gülüyordu içlerinden.',
    'Eski beyefendiler kalmadı artık. Eskiden mahallede gerçekten kibar adamlar vardı. Selam verirler, başlarını eğerlerdi. Şimdi kimse kimseye merhaba bile demiyor. Devir değişti, insanlar bozuldu.',
]

print(f'Huseyin Rahmi Gurpinar: {len(HUSEYIN_RAHMI)} metin')

# Yazar 3: Halit Ziya Usakligil (1866-1945)
# Servet-i Funun, suslu dil, konak/yali, melankoli
HALIT_ZIYA = [
    'Salonun loş ışığında her şey bir rüya bulanıklığında görünüyordu. Ağır kadife perdeler camlardan içeri sızan akşam güneşinin son ışıklarını süzüyordu. Mobilyalar gölgelerle birleşip belirsiz silüetler hâlinde duruyordu.',
    'Konağın geniş avlusu çiçeklerle doluydu. Yasemin kokuları, gül kokuları, akşam serinliğinde havayı doldurmuştu. Mehtap doğmuş, etrafa o gümüşi parıltısını yaymıştı. Genç kız balkonda durmuş, uzaklara dalmıştı.',
    'Köşkün geniş kütüphanesi, eski kitapların ağır kokusuyla dolup taşıyordu. Yıllardır kimse bu kitapları açmamıştı. Onun parmakları cilt üzerinde gezindi, sayfaları kaldırırken sanki bir mazi parçasını uyandırıyor gibiydi.',
    'Sahildeki köşk, akşamın ilk gölgeleriyle örtülürken denizde son vapurların ışıkları yanıp sönüyordu. Pencereden bakanın ruhunda bir hüzün uyanırdı, fakat bu öyle tatlı, öyle ince bir hüzündü ki, kimse ondan kurtulmak istemezdi.',
    'Akşamleyin teras kapısı açıldığında, denizin kokusu salonu doldurdu. Misafirler havadan, sudan bahsediyorlardı. Onun gözleri ise uzaklara, ufkun erimiş yerlerine takılı kalmıştı.',
    'Bahçedeki ağaçlardan biri, sonbahar rüzgârıyla son yapraklarını döküyordu. Genç kadın camın ardından bunu seyrediyordu. Hayatının da böyle bir yapraklar dökümü içinde olduğunu hissediyor, içinde tarif edilmez bir keder uyanıyordu.',
    "Yalının üst katından Boğaz'ı seyreden balkonda iki sessiz kişi oturuyordu. Aralarında geçen sohbet bitmiş, şimdi sadece deniz konuşuyordu. Dalgalar yalıya çarpıyor, akşamın o tarif edilmez melalini taşıyordu.",
    'Eski konağın merdivenleri yıllarca pek çok ayağı taşımıştı. Şimdi onları çıkan ayaklar daha hafifti, daha ağırbaşlıydı. Sanki konak hatıraların ağırlığı altında biraz daha yaşlanmıştı.',
    'Salondaki kristal avizenin altında, yıllar önce nice baloların yapıldığı söylenirdi. Şimdi mum yakılmıyordu artık, salondaki sandalyeler boş kalmıştı. Her şey eski, her şey hatıraların solgun gölgesinde.',
    'Konağın bahçesinde küçük bir havuz vardı. İçinde nilüferler büyürdü yaz aylarında. Genç kadın saatlerce başında oturur, suyun yüzüne kendi yüzünü düşürürdü. Hayatın akışı gibi geliyordu ona o sular.',
    'Yalının deniz tarafındaki odasından akşamleyin uzakta vapurların ışıklarını seyrederdi. Her ışık başka bir şehre, başka bir hayata gidiyordu. O ise yerinde, sandalyesinde, hep aynı odada, hep aynı bakışla.',
    'Akşam ışığı odanın penceresinden içeri süzülüyor, halıya soluk bir renk veriyordu. Genç adam koltuğa gömülmüş, gözleri tavanda, düşüncelere dalmıştı. Dışarıda hayat sürüyordu, içeride ise bir mazi uyuklayışı.',
    'Konağın bahçe duvarları yüksekti. İçeride bir dünya, dışarıda başka bir dünya vardı. İçeridekiler dışarısını anlamazlardı, dışarıdakiler içerisini. Bu iki dünya bir kavak ağacının iki yanı gibi yan yana dururdu, asla karışmazdı.',
    'Köşkün penceresinden mehtap o kadar güzel görünürdü ki, içeride kimse uyumak istemezdi. Saatlerce baş başa otururlar, ayın sulara vuran ışığını seyrederlerdi. Hayatın bütün gürültüsü o anda uzakta kalırdı.',
    'Misafirler salonda oturmuş, çay içiyordu. Hostes Bayan asalet ve zarafetle dolaşıyordu masalar arasında. Her birine ayrı bir söz, ayrı bir tebessüm. Hepsi onun zarif terbiyesinden etkileniyordu.',
    "Yalının taraçasında akşamlar çabucak inerdi. Güneş Marmara'ya yavaş yavaş gömülürken, ufukta pembe-mor bir tablo oluşurdu. İhtiyar hizmetçi sessizce mumları yakar, salona buyur ederdi konukları.",
    'Genç kadın, aynanın karşısında dakikalarca durdu. Saçlarını topladığı zaman boynunun zarafetini, gözlerindeki o derin hüznü daha iyi görüyordu. İçindeki yangın dışarıya hiçbir surette aksetmiyordu, herkesten saklanmış bir sırrı vardı.',
    'Onun sözleri genç adamın kalbinde acı bir damla gibi yer etti. Yıllardır sakladığı, kimseye söyleyemediği duygular bir anda alt üst olmuştu. Salonun lüks dekoru içinde, herkesin gözü önünde, kendini yapayalnız hissediyordu.',
    'Suskunluğun mahiyeti müphemdi, kelimeler yetmiyordu hisleri ifadeye. Akşam yemeğinde annesinin pırıl pırıl sofrasında, gümüş şamdanların ışığında, herkes neşeliydi. Yalnız o bir köşede, sessizce yemeğini yiyordu.',
    'Solgun bir tebessüm, bir hatıranın yansıması gibi belirdi yüzünde. Yıllar önce yaşadığı bir aşk hatırlanmıştı. O zamanlar her şey başkaydı, o gence, dünya bambaşkaydı, fakat şimdi her şey solmuş, soluk bir hâle bürünmüştü.',
    'Sevdiği kadının yüzünde o günden beri bir solgunluk vardı. Kimse sebebini sormadı, kimse anlamak istemedi. Etrafındakiler bencildiler, kendi dertleriyle meşguldüler. O da bu suskun ıstırabı yalnız taşımayı yeğledi.',
    'Genç adamın kalbi hızla atıyordu, fakat dışarıya hiçbir şey belli etmiyordu. Tebessümünde bir samimiyet vardı, gözlerinde ise yıllarca biriken hüzün. Sevgilisine bakarken her şeyin geç olduğunu anlıyor, bir damla yaş gözünde donup kalıyordu.',
    'Kalbinde bir yangın vardı, fakat etrafındakilerin hiçbiri farkında değildi. Salonun ortasında, ışıkların altında, kahkahalar arasında o kendi içine kapanmıştı. Belki bütün hayatı boyunca böyle olmaya devam edecekti.',
    'Aşk, onun için bir hapishane olmuştu. Sevdiğinden vazgeçemiyor, fakat ona sahip de olamıyordu. Bu ikilik içinde günler geçiyor, yıllar geçiyor, o solgunlaşıyordu. Aynaya baktığında kendini tanımıyordu artık.',
    'Genç kadın o tablonun karşısında durmuştu yine. Tabloda eski bir manzara vardı, fakat ona göre çok daha fazla şey ifade ediyordu. Onun ilk aşkını yaşadığı şehir bu manzaraya benziyordu. Şimdi sadece bir hatıra olarak kalmıştı.',
    'Aşk-ı memnu denilen o uçurumun kenarına geldiğinde durup kaçabilirdi. Fakat o duramadı, kaçmadı. Yıllar sonra geriye baktığında pişman olacaktı, ama pişmanlık bile aşkın bir parçasıydı.',
    'Onun bakışlarında bir tehlike vardı. Yıllarca kendini tutmuştu, fakat o gece bir şey değişmişti. Bahçede yalnızdılar, ay ışığı ağaçların arasından sızıyordu. Konuşmadan dakikalarca durdular, sonra o ilk adımı attı.',
    'Genç kadının gözlerinde yaş birikti, fakat hiçbiri yanağına düşmedi. İçinden ağlayışın anlamı vardı, gözyaşı dökmek bayağılıktı. Onun hüznü zarif olmalıydı, daima, hayatın her anında.',
    'İki sevgili konak bahçesinde sessizce yürüyordu. Aralarında geçen sözler azdı, fakat her bakış bir cümle yerini tutardı. Bu zarif aşkın sırrını yalnızca onlar bilirdi, etrafındakiler ise akşam yemeği muhabbetinde meşgullerdi.',
    'Yıllar onun yüzüne biraz çizgi düşürmüştü, fakat hüznünü artırmaktan başka bir şey yapmamıştı. Eskiden bir tebessüm odasını aydınlatırdı, şimdi aynı tebessüm soluk bir solgunluğa dönüşmüştü. Bunun farkında olan tek kişi ise oydu.',
    'Aşkı tarif etmek istese, kelimeler yetmezdi. O öyle bir şeydi ki, yaşandıkça anlaşılırdı, anlatılmazdı. Belki şiir biraz tarifini yapabilirdi, belki müzik. Fakat sözler kifayetsizdi, hep eksik kalırdı.',
    'Ruhunda bir tahassür, bir yorgunluk, hayatın bütün ağırlığı omuzlarına çökmüş gibi bir his vardı. Yazmak istediği şiir, kelimeler arasından kaçıp gidiyordu. Kâğıdı önüne almış, mürekkeple parmaklarını lekeletmişti, fakat hiçbir mısra dökülmüyordu kaleminden.',
    'Hayatın bir anlamı kalmamıştı sanki. Eskiden her şey ona zevk verirdi, kitap okumak, müzik dinlemek, dostlarla konuşmak. Şimdi hepsi soluk birer hatıra olmuştu. İçinde bir boşluk vardı, hiçbir şey o boşluğu dolduramıyordu.',
    'İçindeki o yangın kimseye anlatılamazdı. Etrafındakiler onu hep neşeli, sakin, terbiyeli görürlerdi. Halbuki kalbinde fırtınalar koparıyordu. Bu zıtlık onu yıpratıyor, yavaş yavaş eritiyordu.',
    'Bazen aklından geçenleri yazmaya çalışırdı, fakat kâğıt boş kalırdı. Kelimeler kendinden kaçar, gizlenirdi sanki. Hisleri o kadar inceydi ki, kabasından kelimeler onları taşıyamazdı. Bu yüzden hep susardı, hep yazamazdı.',
    'Akşam saatleri ona ayrı bir hüzün verirdi. Güneş çekilirken, gölgeler uzarken, ruhuna bir ağırlık çökerdi. Bunun sebebini bilmezdi, fakat her akşam tekrar olurdu bu his. Sanki içinde bir şey ölürdü her gün batımında.',
    'Onun melali bir hastalık değil, bir asalet alameti idi. Yıllarca süren ince bir terbiyenin sonucu olan bu hüzünlü ruh hâli, ona ayrı bir zarafet kazandırırdı. Fakat aynı zamanda onu yıpratan, kemiren bir şeydi.',
    'İnce duygulu insanlar her zaman acı çekerlerdi. O da bunlardan biriydi. Her sözcük onun kalbinde derin bir iz bırakırdı, her bakış bir damla daha yaşa neden olurdu. Yaşamak onun için ağır bir yüktü, fakat ölmek de o kadar uzak.',
    'Geceleri uyuyamadığı olurdu çoğu zaman. Salonda otururdu, lambanın ışığında. Kitap açar, fakat okuyamazdı. Düşünceler kalbine sökün ederdi, geçmiş ve gelecek karışırdı zihninde.',
    'Düşünceleri bir nehir gibi akıyordu, durdurulamayan, yönlendirilemeyen. Her gece aynı düşünceler dönüyordu kafasında. Eski hatıralar, yarım kalmış sözler, yapılmamış işler. Sabaha kadar bu nehir akmaya devam ederdi.',
    'Hayata karşı bir küskünlük vardı içinde. Onu yetiştirenler her şeyi vermişlerdi, sevgi, terbiye, mal. Fakat o yine de tatmin olamıyordu. Hep bir eksik vardı, hep bir özlem. Belki de bu insanın doğasından geliyordu.',
    'Genç adamın yüzünde sürekli bir tasavvur ifadesi vardı. Olmuş şeyleri unutamıyor, olacak şeyleri tahayyül edemiyordu. Şimdi ile geçmiş arasında, gerçek ile hayal arasında, bir yerde duruyordu.',
    'Bahçedeki sandalyede oturmuş, çiçekleri seyrediyordu. Bir kelebek konup kalkıyordu güllerin üzerine. O kelebek gibi olmayı dilerdi bazen, hiçbir derdi olmadan uçabilmeyi, fakat insan olarak doğmuştu, ağır bir ruhu taşıyordu.',
    'Eski mektupları çıkardı çekmeceden, birer birer açtı. Hepsi yıllar önce yazılmıştı. Yazanlardan bazıları artık yoktu, bazıları çok uzaklarda. Mektupları okurken bir damla yaş döküldü kağıdın üzerine, mürekkep yayıldı.',
    'Kontesin verdiği baloda, salonun ortasında valsler dönüyordu. Genç kadınların etekleri parlıyor, beyefendiler nazik adımlarla onlara eşlik ediyorlardı. O ise bir köşede oturmuş, bütün bu zevki uzaktan, kederli bir tebessümle seyrediyordu.',
    'Akşam yemeği müsterhi bir atmosferde geçti. Misafirler nezaketle konuşuyor, hizmetçiler sessizce servis yapıyordu. Yemekler nefisti, masa örtüsü kar gibi beyazdı. Her şey kusursuzdu, fakat onun ruhunda hiçbir karşılığı yoktu.',
    'Hanımefendi misafirleri salona buyur etti. Hepsi otururken zarafetlerinden taviz vermediler. Hizmetçiler kahve servisi yaptı, hanımlar fincanları ince parmaklarıyla tuttu. Aralarındaki muhabbet o kadar inceydi ki, dışardan biri anlamazdı.',
    'Konak balosunda her şey hazırdı. Kristal avizeler yanıyor, salonun parkesi cilalanmıştı. Misafirler birer birer geliyor, ev sahibesi onları kapıda karşılıyordu. Akşam saatleri ilerledikçe salon doluyordu.',
    'Mösyö ve madam zarif bir çift oluşturuyordu. Onlar salona girdiğinde gözler hep onlara çevriliyordu. Hareketleri ölçülü, sözleri seçilmiş, tebessümleri zarif. İdeal bir asaletin temsilcileriydiler.',
    'Misafirlerin gevezelikleri salonu doldurmuştu, fakat o duymuyordu hiçbirini. Pencerenin yanına gitmişti, ay ışığı denizde parlıyordu. O görüntü ona kelimelerden daha çok şey söylüyordu.',
    'Balonun en önemli misafiri kontes hazretleriydi. Salona girdiğinde herkes hazır ola geçti. Etekleri bir bahar çiçeği gibi açılmış, başında zarif bir taç. Yanından geçerken hafif bir parfüm dalgası yayılırdı.',
    'Salonda piyano çalan genç hanım son derece muhteremdi. Parmakları tuşlar üzerinde uçuyor, müzikten zarif bir su gibi sızıyordu. Misafirler dinlerken pencerelerden ay ışığı süzülüyordu, atmosfer büyülüydü.',
    "Mösyö Avrupa'dan dönmüştü, salonda son moda kıyafetler giyiyordu. Hanımlar onu ilgiyle dinliyor, Paris'ten anlattığı şeyleri merakla takip ediyorlardı. Onun her sözünde bir incelik, bir zarafet vardı.",
    'Yemek masasında sohbet zarif konulara dair sürüyordu. Edebiyat, müzik, son operalar. Hizmetçiler tabakları sessizce değiştiriyordu. Mum ışığında her yüz biraz daha güzelleşiyor, gözler daha derin görünüyordu.',
    'Çay saati salonun en sevilen vaktiydi. Hanımefendi gümüş takım takımları çıkartırdı, fincanlar Limoges porseleninden. Hanımlar misafirleri buyur eder, çayın yanına nefis pastalar konurdu. Üç dört saat geçerdi farkına bile varmadan.',
    'Akşamleyin yalıya konuk gelmişti. Tanınmış bir şair, ailenin eski dostuydu. Salonda otururken şiirlerinden okumaya başladı. Sesi tatlı, kelimeler süslüydü. Misafirler büyülenmişçesine dinliyordu, dışarıda mehtap yükseliyordu.',
    'Hanımefendi her perşembe akşamı salonunu açardı. Şehrin en seçkin kişileri toplanırdı orada. Şairler, ressamlar, müzisyenler. Sanat üzerine konuşulur, fikirler tartışılırdı. Bu salonların değeri paha biçilmezdi.',
    'Yeni gelen misafir konak hayatına alışkın değildi. Salona girdiğinde hareketleri biraz tutuktu, sözleri biraz kabaydı. Hanımefendi onu nazikçe yönlendirdi, hiç kimse fark etmedi alaylı bakışları.',
    "İstanbul'un yağmurlu günleri ona ayrı bir keder verirdi. Pencerenin önünde durur, camlardan akan damlaları seyrederdi. Sanki şehir de onunla beraber ağlıyordu. Bu hüzün ona tanıdık geliyor, neredeyse tatlı geliyordu.",
    'Boğaziçi sularının üzerinde akşam puslu bir tülle örtülüyordu. Vapurların ışıkları uzaklarda yanıp sönüyor, balıkçı kayıkları kıyıya yanaşıyordu. Sahildeki yalılardan piyano sesleri yayılıyordu, akşamın o tarif edilmez melali her tarafa hâkimdi.',
    'Sahile inen yol, badem ağaçları arasından geçiyordu. Bahar gelmişti, ağaçlar pembeleşmişti. Genç kız kolunda annesi, yavaş adımlarla yürüyordu. Hava o kadar tatlıydı ki, konuşmaya bile gerek yoktu.',
    'Yalının önündeki iskelede vapur duruyordu. Hizmetçiler bavulları yüklüyor, hanımefendi son talimatları veriyordu. Konak yazlığa gidiyordu, mevsim böyleydi. Üç gün önceden hazırlıklar başlamıştı.',
    "Boğaz'da bir bahar akşamıydı. Hava o kadar berraktı ki, karşı yakadaki köşkler tek tek görünüyordu. Boğaz'ın iki yakası birbirine selam veriyordu sanki, akşam ışığı her şeye ayrı bir güzellik veriyordu.",
    'Sahile inen merdivenler eskiydi, taşları yer yer kırılmıştı. Yıllarca pek çok ayak basmıştı oraya, yıllarca konak hayatının bir parçası olmuşlardı. Şimdi sessizdiler, sadece dalgalar onlara çarpıyordu.',
    "İstanbul'un sonbaharı melalin tâ kendisiydi. Yapraklar dökülüyor, hava soğumaya başlıyor, akşamlar erken iniyordu. Konak pencerelerinden bakan herkes içine işleyen bir hüzün hissederdi bu mevsimde.",
    'Sahildeki köşkün arka bahçesinde küçük bir gül bahçesi vardı. İhtiyar bahçıvan onlara her gün bakıyordu. Bahar geldiğinde güller açar, yaz boyunca kokuları havayı doldururdu. Sonbaharda yapraklar dökülürdü, bahçe yine sessiz olurdu.',
    "Köşkten Boğaz'a bakan bir balkon vardı. Sabah erkenden oraya çıkar, gün doğumunu seyrederdi. Güneş Anadolu yakasından doğar, sular önce pembeleşir, sonra altın rengini alırdı. Bu manzara onun en büyük zevkiydi.",
    'Yağmurlu bir günün ardından İstanbul mistik bir görünüm alırdı. Camlar buğulanır, sokaklar parıldardı, hava temizlenirdi. Yalının camında damlalar süzülürken, içeride sıcak bir soba yanardı. Manzara şiir kadar güzeldi.',
    'Akşamleyin denize inen merdivenden hizmetçi kız çıktı. Elinde çamaşır sepeti vardı, sahilde yıkayacaktı. Suların üzerinde gümüşi bir ay ışığı dalgalanıyordu. Onun saçları rüzgârda uçuşuyor, basit hâliyle bile zarif görünüyordu.',
    "Marmara'nın üzerine pus inmişti. Vapurlar düdükleriyle birbirlerine seslenirken, ufukta belirsiz silüetler dalgalanıyordu. Yalı sakinleri pencereye toplanmıştı, bu manzaranın bir resmini bekliyorlardı sanki.",
    'Yaşlı kalfa konağa yıllar önce gelmişti. Hanımefendinin gençliğini bilirdi, onun bir oğlunu doğurduğunu, sonra başka bir doğumda kaybettiğini hatırlardı. Konağın tüm sırlarını bilen oydu, fakat hiç kimseye söylemezdi.',
    'Hizmetçi kız sessizce salona girdi, masaya kahve servisi yaptı. Hanımefendi başını kaldırıp gülümsedi, kız da hafifçe baş eğdi, çekildi. Bu zarif sahne her sabah tekrarlanırdı, hiç değişmeden.',
    'Aşçı kadın mutfakta bir senfoni şefi gibi yönetiyordu işleri. Tencerelerden buharlar yükseliyor, tatlı kokular bütün konağa yayılıyordu. Akşam yemeği için en az on çeşit hazırlıyordu.',
    'Konak bahçıvanı yıllarca tek başına çalışmıştı. Bahçeyi gözünden sakınır, çiçekleri çocuk gibi büyütürdü. Hanımefendi onun emeğini takdir eder, her bayramda hediyeler verirdi. O da konağa o kadar bağlıydı ki, ayrılmayı düşünemezdi.',
    'Genç hizmetçi kız konağa yeni gelmişti. Konak hayatına alışmakta zorlanıyordu. Yaşlı kalfa onu sabırla yönlendirdi, hata yaptığında kızmadı, sadece doğrusunu gösterdi. Birkaç ay sonra kız konağa adapte olmuştu.',
    "Mösyö Andre Fransız bir mürebbiyi getirmişti çocuklara. Genç kadın Paris'ten gelmişti, kibar ve dindar. Konak halkı onu hemen sevdi, çocuklara Fransızca öğretirken, kendisi Türkçe öğreniyordu yavaş yavaş.",
    'Konağın seyisi atları gibi olmuştu. Onlarla konuşur, onları seyranlara çıkarırdı. Beyefendi sabah atına bindiğinde, seyis hep yanında olurdu. Onun olmadığı bir hayatı düşünemezdi atlar.',
    'Yaşlı uşak yıllarca konakta hizmet etmişti. Beyefendinin babasını da bilmişti gençliğinde. Şimdi torunlara hizmet ediyordu, fakat geçmiş günleri anlatırken gözleri parlardı, sanki gençlik geri gelirdi.',
    'Genç kadının yüzü bahar gelir gelmez solmaya başlamıştı. Doktorlar geliyor gidiyordu, hiçbiri çare bulamıyordu. Hanımefendi gece sabahlara kadar başında otururdu, hastalığı bir türlü teşhis edemiyorlardı.',
    'Yaşlı beyefendi koltuğunda otururken gözleri uzaklara dalmıştı. Yıllarca yaşamış, çok şey görmüş, çok şey bilmiş bir adamdı. Şimdi son günlerini sayıyordu, fakat hiçbir korku yoktu yüzünde, sadece bir hüzün.',
    'Hastalığın ağırlaştığı günlerde konak sessizleşmişti. Herkes sessizce konuşuyor, hizmetçiler ayaklarının ucuna basarak yürüyordu. Hastanın odası karanlıktı, pencereler kapalıydı. Doktor başında nöbet tutuyordu.',
    'Genç şair veremden gidiyordu yavaş yavaş. Bunu biliyordu, kabullenmişti bile. Son şiirlerini yazıyordu, kalbini kâğıda dökerek. Sevdiği kadın başında oturuyor, kalemi titreyen elinden alıp yazıyordu.',
    'Eski konağın hanımı bir gün yatağa düşmüştü. Yıllarca hâkim olduğu evi artık göremiyordu. Doktorlar geliyor, ona umut veriyordu, fakat o biliyordu ki son geliyordu. Sessizce, asaletle bekliyordu.',
    'Ölüm konağa girdiğinde, bütün hizmetçiler pencereleri kapadı. Aynalar siyah örtülerle örtüldü. Akşam müezzinleri okuyacaktı, hayır duaları yapılacaktı. Konak yıllarca daha matem giyecekti.',
    'Cenaze konaktan çıkarken bütün mahalleli yollara dökülmüştü. Onun bu kadar sevildiğini kimse bilmiyordu. Etrafa hayır işleri yapmış, fakirlere yardım etmişti gizliden gizliye. Şimdi herkes onu anıyordu.',
    'Yaşlı beyefendi balkonun en köşesinde oturmuş, eski günleri hatırlıyordu. Gençlik yılları, ilk aşkları, kaybettiği dostları. Hepsi onun gözlerinin önünden bir film şeridi gibi geçiyordu. Bir damla yaş yanağından süzüldü.',
    'İhtiyar hanımefendi çekmecesinden eski bir mektup çıkardı. Yıllar önce sevdiği genç adamdan gelmişti. Kâğıt sararmış, mürekkep solmuştu. Onu bir daha okudu, başka biri yazmış gibi. Belki de gerçekten başka biri yazmıştı, çünkü o genç kız artık yoktu.',
    'Konağın eski albümleri sandığın dibinde saklıydı. Açıldığında zaman geriye dönerdi sanki. Sararmış fotoğraflar, gülümseyen yüzler. Çoğu artık hayatta değildi. Yaşlı hanım birer birer bakıyordu, her yüze ayrı bir hatıra.',
    'Yıllar geçmişti, fakat o sahil ona hâlâ eski hatıraları getiriyordu. Kırk yıl önce ilk aşkıyla orada yürümüştü. Şimdi yalnızdı, fakat o yürüyüş hâlâ canlıydı zihninde. Hatıralar hayatından daha gerçek geliyordu artık.',
    'Eski piyano salonun bir köşesinde sessizce duruyordu. Yıllarca çalınmamıştı, akortsuz olmuştu belki. Fakat yaşlı hanım onu hatırlardı, gençken her gün başında oturduğu o piyanoyu. Bir gün belki dokunurdu yine tuşlarına, fakat o gün gelmiyordu.',
    'Beyefendi pencerenin önünde durmuş, dışarıda oynayan torunlarını seyrediyordu. Kendi çocukluğunu hatırlıyordu. Aynı bahçede oynamıştı yıllar önce, aynı ağacın altında uyumuştu. Hayat dönüp dolaşıp aynı yerlere geliyordu.',
    'Yaşlandıkça insan daha çok geçmişte yaşıyordu. Şimdiki zaman silikleşiyor, eski yıllar canlanıyordu. İhtiyar beyefendi bunu biliyordu, kabullenmişti. Geçmiş onun gerçek vatanıydı artık.',
    'Konağın hizmetçileri yıllarca değişmiş, fakat hanımefendi hep oradaydı. Yıllar onun saçlarını ağartmış, yüzüne çizgiler düşürmüştü. Fakat ruhundaki o zarafet, o asalet hiç değişmemişti, hatta artmıştı.',
]

print(f'Halit Ziya Usakligil: {len(HALIT_ZIYA)} metin')

# Yazar 4: Resat Nuri Guntekin (1889-1956) - 2026'da kamu mali oldu
# Anadolu, tasra, ogretmenlik, romantik dram, sade duygusal Turkce
RESAT_NURI = [
    'Trenden indiğimde küçük kasaba istasyonu boştu. Bir ihtiyar bekçi beni karşıladı, gözleri uykuluydu. Bavullarımı kendim taşıdım, sokakların sonundaki misafirhaneye doğru yürüdüm. Yol uzun, hava sıcaktı.',
    'Kasabanın tek caddesi tozluydu. İki tarafına dizilmiş alçak evler, kepenkleri inik dükkânlar. Her şey uykudaydı sanki. Sadece bir kahveden gelen yüksek sesler duyuluyordu, içeride birkaç adam sesli sesli tartışıyordu.',
    'Pazar günleri kasabanın meydanı dolup taşardı. Köylüler etraf köylerden gelir, mallarını sergilerdi. Tavuklar, peynirler, yumurtalar, sebzeler. Çocuklar etrafta koşuşur, kadınlar pazarlık yapardı. Akşama doğru meydan boşalır, kasaba yine sessizliğine bürünürdü.',
    'Memleketin küçük bir kasabasına atanmıştım. İlk gün otobüsten indiğimde ne yapacağımı bilmiyordum. Etrafta kimse yoktu, sadece bir köpek bana havlıyordu. Çantamı sürükleye sürükleye sokakların içine girdim.',
    'Kasabanın akşamları çok güzel olurdu. Güneş batarken dağların ardından kırmızı bir ışık yayılır, bütün evler bu ışıkla bir an kızıllaşırdı. Sonra hava soğumaya başlar, sokak lambalarının altında sayılı insan görünürdü.',
    'Buralarda yaşamak ilkin zor gelir, fakat sonra alışılır. İnsan kendisini dünyadan ayrılmış hisseder, fakat bu ayrılık zamanla huzura dönüşür. Şehirdeki o telaştan, o gürültüden uzakta, kendiyle baş başa kalmak güzeldir.',
    'Kasaba küçüktü, herkes herkesi tanırdı. Sokakta yürürken her iki adımda bir biriyle selamlaşırdım. Akşamları kahvede oturup memleket meselelerini konuşurduk. Yıllar sonra düşününce, en güzel günlerimi orada geçirmişim.',
    'Otobüs durağında bir köylü bekliyordu. Yanına gittim, yol sordum. O elindeki çayını içerken bana güzelce anlattı. Sonra konuşmaya başladık, bir saat geçti. Adam beni evine çağırdı, ben de gittim, akşam yemeğini birlikte yedik.',
    'Memlekete döndüğümde her şey değişmişti. Bildiğim sokaklar yoktu artık, eski dostlarımdan çoğu ya ölmüş ya başka şehre göç etmişti. Yine de bir hisle eski mahallemde dolaştım, çocukluğumun izlerini aradım.',
    "Anadolu'nun her köşesinde bir hikâye saklıdır. Köy meydanından geçerken bir ihtiyar dede ile konuştuğumda otuz yıl öncesini anlattı. Köyün nasıl yandığını, nasıl yeniden kurulduğunu, kimlerin geldiğini, kimlerin gittiğini. Saatlerce dinledim, bitmek bilmedi.",
    'Kasabanın eczanesi çarşının ortasındaydı. Eczacı orta yaşlı, gözlüklü bir adamdı. Bütün gün dükkânının önünde otururdu, bir kitap okurdu. Hastalar gelirdi, ilaçlar verirdi. Akşam dükkânı kapatır, sokakta yürüyüşe çıkardı.',
    'Hanın avlusunda bir at arabası duruyordu. Atı yere kapaklanmış, dinleniyordu. Sürücü merdivenin başında oturmuş, çayını içiyordu. Onun yanına gittim, kasabaya nasıl gidileceğini sordum. "İki saat sürer" dedi, "hemen yola çıkalım."',
    'Köy meydanında bir çeşme vardı. Sabah erkenden kadınlar başına toplanırdı, su doldurmaya gelirlerdi. Her bakraç dolarken o günkü mahalle haberleri konuşulurdu. Akşamleyin de erkekler gelirdi, abdest alır, camiye giderdi.',
    'Kasabanın hâkimliğine atandığımda yirmi sekiz yaşındaydım. Yeniydim mesleğe, biraz korkuyordum. Fakat ilk günden itibaren mahallenin yaşlıları beni misafir etti, ekmek paylaştı. Bu sıcaklık hâlâ aklımdan çıkmaz.',
    'Akşamleyin sokağa çıkıp yürüyorum. Kasaba uyuyor, sadece köpekler havlıyor. Pencerelerden gelen ışıkları sayıyorum, her birinin ardında bir aile var. Onların hayatlarını hayal etmeyi seviyorum, çocuk gibi bir merak duyuyorum bu evlere.',
    'Sınıfa girdiğimde otuz çocuk başlarını çevirdi, sessizce bana baktılar. Hepsinin gözünde merak vardı, yeni hoca neye benzeyecek diye. Tebessümle selam verdim, bir tanesi ayağa kalkıp "Hoş geldiniz hocam" dedi. Yüreğim sıcacık oldu o an.',
    'Mektepte ilk dersimi verirken çok heyecanlıydım. Çocukların gözlerinde dikkat ve sevgi okuyordum. Anlatmaya başladım, kelimeler kendiliğinden döküldü ağzımdan. Ders bitince çocuklar etrafıma toplandı, sorular sordu. O an öğretmenlik mesleğine âşık oldum.',
    'Köy mektebinin penceresi geniş, ışık boldu. Çocuklar sıralarda otururken kalemler kâğıt üzerinde gezinirdi. Ben tahtaya bir şey yazar, onlar defterlerine geçirirdi. O sessiz odadaki çalışma sesi en sevdiğim müzikti.',
    'Bizim mektebimiz küçüktü, üç odadan ibaretti. Bir tek hoca, yani ben vardım. Sabah birinci, öğleden sonra ikinci sınıfa ders veriyordum. Akşamları evime giderdim, sabah erkenden yine mektepteydim. Bu hayata alışmıştım, sevmiştim.',
    'Çocukların biri annesi olmadan büyümüştü, babası savaşta ölmüştü. Yetimdi, akrabaları onu büyütüyordu. Mektebe geldiğinde elbiseleri yamalıydı, ama gözleri parlardı zekâdan. Onu özellikle severdim, fazla zaman ayırırdım.',
    'Yeni öğretmen olarak gönderildiğim köyde, mektep yıkık dökük bir odaydı. Sıralar kırık, tahta yıpranmıştı. Çocuklar yerde oturuyordu. Köyün muhtarına gittim, durumu anlattım. Birlikte çalıştık, üç ayda mektebi yeniledik.',
    'Mektep tatilleri çocuklar için sevinçti, benim için hüzün. Üç ay boyunca onları görmeyeceğim diye üzülürdüm. Köy meydanında karşılaştığımızda, onları her zaman büyümüş bulurdum. Birkaç ay büyük fark yaratırdı çocuklarda.',
    'Akşamları çocukların defterlerini düzeltirdim. Her birinin yazısı, hatası ayrıydı. Bir tanesi çok güzel resim yapardı kenarına, ben ona bir kalem hediye etmiştim. Bunu hâlâ kullandığını duyduğumda kalbim sevinmişti.',
    'Çocuklara şiir okumayı çok severdim. Anlamasalar bile dinlerlerdi, ezberlerlerdi. Bir tanesi sonradan büyüdüğünde gelmiş, hâlâ o şiirleri ezbere okuduğunu söylemişti. Öğretmenlik mesleği böyle güzeldir, yıllar sonra meyveler verir.',
    'Mektepten döndüğümde yorgun olurdum, fakat memnun. Çocuklar bana enerji verirdi, gençleştirirdi. Onlar olmasa bu kasabada yaşayamazdım. Onlar benim hayatımı dolu hâle getiriyordu, bunu hep söylerdim.',
    'Sınıfta bir kız çocuğu vardı, çok zekiydi. Ondan büyük şeyler olacağını biliyordum. Annesi babası onun okumasını istemiyordu, evde işe yardım etsin diyorlardı. Birkaç kez aileyle konuştum, sonunda ikna ettim. Şimdi öğretmen oldu o kız.',
    'Köy mektebinin avlusunda bir dut ağacı vardı. Yaz başında çocuklar ona tırmanır, dut yerdi. Ben karışmazdım, sadece dikkatli olmalarını söylerdim. Bir keresinde bir tanesi düştü, kolu kırıldı. Onu kasabaya götürdüm, ailesi bana çok teşekkür etti.',
    'Müzik dersi yoktu programda, ama ben çocuklara şarkı öğretirdim arada. Türküler söylerdik birlikte. Onların temiz sesleri sınıfta yankılanırdı. Pencereden geçen yetişkinler durup dinlerdi, sonra gülümseyerek yollarına devam ederlerdi.',
    'Mektep müsamerelerinde çocukların heyecanı görülmeye değerdi. Kostümlerini hazırlamak, rollerini ezberlemek, sahneye çıkmak. Aileleri salonda toplanırdı, herkes kendi çocuğunu beklerdi. Bayram havası eserdi mektepte günlerce.',
    'Mezun olan öğrencilerimden bazıları yıllar sonra geri dönerdi. "Hocam" diye seslenir, ellerimi öperlerdi. Onları büyümüş, mesleğe atılmış görmek beni hem mutlu ediyor hem yaşlandırıyordu. Zaman ne çabuk geçmişti.',
    'Onu ilk gördüğüm gün kasabaya yeni gelmiştim. Çeşme başında duruyordu, su dolduruyordu. Saçları rüzgârda dağılmıştı, yüzünde tatlı bir gülümseme vardı. O an kalbim çarpmaya başladı, fakat kendimi tutmaya çalıştım.',
    'Onunla nehir kıyısında konuşurduk uzun uzun. Akşam üstleri o eve dönmeden önce on dakikamız olurdu. Bu on dakika bütün günümün en güzel anıydı. Yıllar geçti, hâlâ o anlar gözümün önündedir.',
    'Sevdiğim insandan ayrı düşmek hayatımdaki en büyük acıdır. Onu unutmaya çalıştım, başaramadım. Yıllar geçti, mesafe büyüdü, fakat içimdeki o his hiç eksilmedi. Belki ölünceye kadar da eksilmeyecek.',
    'Babası bizim evliliğimize razı olmadı. "Sen yabancısın bu kasabada" dedi bana. "Kızımı şehirli birine veremem." Çok kırıldım o gün, fakat anladım onu. Ben gittim, o orada kaldı. Şimdi başkasıyla evli olduğunu duydum.',
    'Akşam üstü çeşmeye su almaya gittim. Onu bulamadım, evdeydi belki. Tek başıma döndüm, içimde bir hüzün. Pencereye çıkıp evine baktım uzun uzun, ışık yanmıştı. İçeride o vardı, bana ulaşamayacağım kadar yakındı.',
    'Mektup yazıyordum ona her hafta. Cevap geliyordu, fakat seyrekleşmeye başladı. Sonra hiç gelmedi. Birkaç ay sonra bir akrabasından öğrendim, evlenmişti başkasıyla. Mektubumu yırttım, bir daha yazmadım.',
    'Genç kızın gözleri yaşardı ayrılırken. "Beklerim seni" dedi titreyen bir sesle. Ben de "Geleceğim" dedim, fakat içimden "Belki" diyordum. İki yıl sonra gittiğimde başkasıyla evlenmişti. Hayat bekleyemeyenleri af etmiyordu.',
    'Onunla son gece yağmur yağıyordu. Bahçedeki sundurmada durup yağmuru izledik. Hiç konuşmadık, sadece elimi tuttu. Sonra ben şehre döndüm, o kasabada kaldı. Hayatın bizi ayırması o kadar saçma geldi ki, hâlâ kabullenemedim.',
    'Yıllar sonra eski memlekete döndüm, onu gördüm. Çok değişmişti, çocukları büyümüştü. Selamlaştık nazikçe, fakat eski o tatlılık yoktu artık. Ne ben o gençtim ne o o kız. Hayat hepimizi başka biri yapmıştı.',
    'İlk aşkımı asla unutamadım. Yirmi yaşımdaydım, o on dokuzdu. Birlikte hayaller kurardık, gelecek planları yapardık. Sonra hayat araya girdi, ayrıldık. Şimdi ikimizin de saçları ağardı, fakat o ilk aşkımdı ve hep öyle kalacak.',
    'Mektebin önünde duruyordu, beni bekliyordu. "Senin için bir şey yaptım" dedi utangaç bir tavırla. Cebinden bir mendil çıkardı, kenarına benim ismimi nakşetmişti. O mendili hâlâ saklıyorum, yıllar geçti, kumaşı sararmış, ama ismim hâlâ duruyor üzerinde.',
    'Aşk bir hastalık gibi geldi bana, beni dipten yedi. Geceleri uyuyamıyordum, gündüzleri yemek yiyemiyordum. Onu görmediğim günler çekilmez oluyordu, gördüğüm günler de bir başka azap. Sevmek bu kadar acı verir miymiş, bilmiyordum.',
    'Sevdiğim adam yıllar sonra geri döndü. Kapımı çaldı, açtım, karşımda duruyordu. Yaşlanmış, beli bükülmüştü. "Geç değil mi?" diye sordu. "Geç" dedim, kapıyı kapadım. Sonra arkasından döndüm pencereye, gözyaşlarımla onun gidişini izledim.',
    'Tren istasyonunda beklerken etrafımdaki insanları seyrediyordum. Herkes bir yere gidiyordu, herkesin bir hikâyesi vardı. Birçoğu yorgun, birkaçı neşeli, çoğu sessizdi. Ben de onların arasında oturmuş, kendi yolumu bekliyordum.',
    "Üç gün önce vapurla yola çıkmıştım. Şimdi tren beni Anadolu'nun kalbine götürüyordu. Pencereden bakıyorum, manzara değişiyor durmadan. Kasabalar, köyler, ovalar, dağlar. Memleketim ne büyük, ne güzelmiş, hiç böyle düşünmemiştim.",
    'Trende karşımda bir aile oturuyordu. Anne, baba, üç çocuk. Çocuklar yolculuğa çok şaşırmıştı, pencereden hiç ayrılmıyorlardı. Anneleri sandviç dağıtıyordu, baba gazetesini okuyordu. Bir aile portresi gibi mutlu görünüyorlardı.',
    'Vapurun güvertesinde durmuştum, denizi seyrediyordum. Karşıma genç bir adam geldi, yanımdaki yere oturdu. Aramızda söz kalmadı uzun süre. Sonra o başladı konuşmaya, hayat hikâyesini anlattı. Bir saat geçtiğinde dost olmuştuk.',
    'Otobüs durağında bekliyorum saatlerce. Otobüs gecikmiş, bir mesele varmış sonradan duydum. Çantamı yere koydum, üzerine oturdum. Karşımdaki bakkaldan çay aldım, sigaramı yaktım. Bu kasabaları seviyorum, sabra alıştırıyor insanı.',
    'Yolculuk insanın aklını boşaltır, ruhunu açar. Ben de bunun için sık sık yolculuğa çıkardım. Bir hafta sonu paketimi alır, bir köye giderdim. Orada üç gün kalır, dönerdim. Bu kısa kaçışlar bana hayat verirdi.',
    'Köprü üstünden geçen tren ürkütücü gürültüler çıkarıyordu. Aşağıda nehir akıyordu, üzerinde küçük bir tekne. Tekneden bir balıkçı bana el salladı, ben de ona salladım. Sonra köprü geçildi, ben tekneyi göremedim artık.',
    'İstasyondaki saat yarım saat gerideydi. Bu yüzden treni kaçırmıştım. Bir sonrakini beklemek için dört saatim vardı. Çantamı bankın altına koydum, kasabayı gezmeye çıktım. Sonradan o dört saat hayatımın en güzel keşiflerinden biri oldu.',
    'Eski bir vapurda yolculuk ediyordum. Boyaları dökülmüş, koltukları yıpranmıştı. Ama içeride büyüleyici bir atmosfer vardı. Yolcular birbirine yakın oturmuş, sohbet ediyorlardı. Sanki hep birlikte yaşadıkları evdi vapur.',
    'Trenin penceresinden sürekli dışarı bakıyordum. Manzara hep değişiyordu, ama bir şey hiç değişmiyordu: insanların çalıştığı tarlalar, koşan çocuklar, otlayan koyunlar. Memleketin değişmeyen yüzü, hep aynı, hep huzur veriyordu.',
    'Çocukken babamla birlikte tarlaya giderdik. O ekmeği bir mendile sarar, kuşağına bağlardı. Yolda anlatırdı bana eski hikâyeleri. Onun yanında olmak, onun gölgesinde yürümek hayatımın en mutlu zamanlarıymış. Ne kadar geç anladım.',
    'Annem her sabah erkenden kalkar, mutfakta sobayı yakardı. Ben sıcaklığa uyanırdım. Yatağımdan kalkıp mutfağa giderdim, o ekmek kızartıyordu. Bana taze ekmek dilimi, sıcak süt verirdi. Bu kahvaltıyı yıllar sonra hâlâ özlüyorum.',
    'Mahallenin çocuklarıyla sokakta oynardık akşamlara kadar. Saklambaç, körebe, çelik çomak. Anneler pencereden seslenirdi sırayla. Akşam ezanından sonra herkes evine girer, sokaklar bir anda boşalırdı.',
    'Babam bana ilk defa yüzmeyi öğrettiği günü hatırlıyorum. Beş yaşımdaydım, korkuyordum sudan. O beni sırtına almıştı, yavaş yavaş suya girmiştik. Sonra bırakmıştı beni, çırpınmıştım. Birkaç gün sonra yüzebiliyordum.',
    'Köyde bir derenin kenarında oynardık. Ayakkabılarımızı çıkarır, suya girerdik. Soğuk olurdu, ama eğlenirdik. Akşam eve döndüğümde annem kızardı çamur olmuş ayaklarıma. Ben gülerdim, o da güler, beni temizlerdi.',
    'Babamın bana bıraktığı tek miras, kitaplarıydı. Üç sandık kitap. Çoğunu o okumuştu, kenarlarına notlar düşmüştü. Onları okuduğumda babamla konuşur gibi olurdum. Çocukluğumdaki o yokluğu o kitaplar tamamladı.',
    'Anneanne her cumartesi torunlarına yemek yapardı. Hepimiz toplanırdık onun küçük evinde. Yer azdı, kalabalık çoktu, fakat o sofralar hayatımın en güzel anılarıdır. O sofralar olmasa belki birbirimizi yılda bir görürdük.',
    'Çocukken hastalandığımda annem başımda nöbet tutardı. Geceleri uyanır bakardım, o yatağın yanında otururdu. Alnıma soğuk bezler koyardı, dua okurdu. Onun o şefkati hayatımda yaşadığım en kıymetli şeydir. Yıllar sonra anladım kıymetini.',
    'Yaz tatilleri köydeki dedemde geçirirdim. Onun bahçesi büyüktü, ağaçlar boldu. Ağaçlara tırmanır, meyve yerdim. Köpeği ile koşardım, kedileri ile oynardım. Şehre döndüğümde bu yazlar bana yeniden bir yıl yaşatırdı.',
    'Babam vefat ettiğinde on iki yaşındaydım. O zaman tam anlayamadım kaybımı. Yıllar sonra anladım, hep eksik kaldı içimde. Hayatımdaki bazı kararları onsuz almak zor oldu, hep "O olsaydı ne derdi" diye düşündüm.',
    'Eski albümümü açıyorum. İlk fotoğraf bir okul fotoğrafı, ben en arkadayım, bir sıra çocuk önümde. Hepsi gülüyor, ben ciddi bakıyorum kameraya. O an ne düşünüyordum, ne hissediyordum, hatırlamıyorum artık. Yıllar her şeyi siliyor.',
    'Anneannem köylüydü, okuma yazma bilmezdi. Fakat onun bilgisi başkaydı, hayat bilgisiydi. Bana her şeyi öğretti, en derin dersleri o verdi. "Sabırlı ol kızım, hayat acımasızdır, ama sen iyi kal" derdi. Sözlerini hep hatırlıyorum.',
    'İlkbahar gelmişti, kasabaya yeni bir hava geldi. Ağaçlar yeşermeye başladı, çiçekler açtı. Ben de ruhumda bir hafiflik hissettim. Pencereyi açıp temiz havayı içime çektim. Hayatın yeniden başlamasıydı bu, her sene tekrarlanan o güzel başlangıç.',
    'Sonbaharın o kendine has hüznü vardı. Yapraklar dökülüyor, hava soğumaya başlıyordu. Köy yolları çamur olurdu yağmurla birlikte. Çocuklar mektebe giderken eski çizmeleri giyerdi, ben sınıfta sobayı yakardım. Sıcak bir kahve hayatımın en güzel anıydı.',
    'Kış geldi, kasaba kar altında kaldı. Yollar kapandı, kimse ev dışına çıkamadı. Sobamın etrafında kitap okudum üç gün boyunca. Pencereden bakıyordum ara sıra, kar lapa lapa yağıyordu. Bu sessizlik bana iyi geldi, bana hep iyi gelirdi.',
    'Köy meydanında bir kavak ağacı vardı, çok eskiydi. Yıllarca ona bakıp büyümüştü mahalleli. Yaz aylarında onun altında oturulurdu, gölgesinde serinlenirdi. Çocuklar oraya tırmanırdı, ihtiyarlar tavla oynardı. O ağaç köyün kalbi gibiydi.',
    'Yağmur yağıyordu sabahtan beri. Pencereden dışarı baktım, sokakta kimse yoktu. Çocuklar mektebe gelemedi belki. Ben tek başıma sınıfa gittim, biraz bekledim, sonra sobayı söndürüp eve döndüm. O yağmurlu gün bana ders vermenin değerini hatırlattı.',
    'Tarladaki başaklar sararmış, hasat zamanı gelmişti. Köylüler erkenden tarlalara dağılmıştı. Ben de bir gün gittim onlarla beraber. Orak salladım, sıcak oldum, terledim. Akşam eve döndüğümde her tarafım ağrıyordu, ama bir gurur duyuyordum.',
    'Köyün kenarında bir gölet vardı. Yaz akşamları oraya giderdim, bir kitap okurdum. Sular durgun, etraf sessiz. Kuşlar uçardı kıyıdan. Bu manzara bana hayatın güzelliğini hatırlatırdı, kasabanın bütün sıkıntılarını unuttururdu.',
    'Bahar gelince köyün yolları yine açılırdı. Kasabayla köy arasındaki at arabaları işlemeye başlardı. Mektubum gelirdi şehirden, eski dostlarımdan. Onları okurken gözlerim yaşarırdı, fakat aynı zamanda sevinirdim, unutulmamıştım.',
    'Sonbahar akşamlarında hava sürpriz şekilde berraktır. Yıldızlar bir başka parlar. Ben balkona çıkıp saatlerce gökyüzüne bakardım. Bu kasabada şehirden daha çok yıldız görünüyordu. Belki de gözlerim daha çok açılmıştı, kim bilir.',
    'Kasabanın etrafı bağlardandı. Eylülde üzümler olgunlaşırdı. Ben de bağ sahiplerinden biriyle dostluk kurmuştum, beni her hasata davet ederdi. Bir gün geçirirdim onun bağında, akşam dolu eve dönerdim.',
    'Köyün muhtarı yaşlı bir adamdı, gözleri yorgun bakardı. Çok şey görmüştü, çok şey biliyordu. Akşamları kahvede oturup eskiyi anlatırdı. Çocukken yaşadığı zorlukları, gençken kavuştuğu sevdayı, yıllar sonra kaybettiği oğlunu. Onu dinlemek bir kitap okumak gibiydi.',
    'Komşu kadın bana her sabah ekmek getirirdi. Tek başıma yaşıyordum çünkü, evimde fırın yoktu. Bedavaya verirdi ekmeği, ben de ona kitap okurdum bazen. Akşam ezanından sonra gelir, on dakika otururdu, sonra evine dönerdi. Bu küçük dostluğu çok seviyorum.',
    'Köyün berberi Ferhat Usta bütün haberleri bilirdi. Dükkânına gelen herkesle konuşurdu, kasabanın nabzını tutardı. Bana her hafta tıraş yapardı, sonra çay ısmarlardı. Onunla yarım saat konuşmak gazete okumakla eşdeğerdi.',
    'Tek başına yaşayan yaşlı bir kadın vardı köyde. Kimse onu ziyaret etmezdi, çünkü kimsesi yoktu. Ben pazar günleri ona giderdim. Çay içerdik, konuşurduk. O bana ekmek verirdi, ben ona kitap okurdum. Bu küçük arkadaşlık ikimizi de mutlu ediyordu.',
    'Çiftçi Halil Bey emrinde çalışan iki yardımcısıyla tarlalarını süren bir adamdı. Sabah erken kalkardı, akşam geç dönerdi. Bir gün beni tarlaya götürdü, çalışmayı bana gösterdi. "Bu toprak bana her şeyimi verdi" dedi. "Çocuklarımı büyüttüm bu toprakla."',
    'Köyün muallimi olmadan önce buraya bir başka hoca gelmiş. Bir yıl kalmış, sonra şehre dönmüş. Köylüler hâlâ onu anardı, beni de onunla karşılaştırırlardı. "Eski hocamız buranın suyunu beğenmemişti" derlerdi. Ben gülerdim, ben her suyu içiyorum diye.',
    'Köyde bir delikanlı vardı, on dokuz yaşında. Beni çok severdi, akşamları gelir benimle konuşurdu. Sonra askere gitti, üç yıl gelmedi. Döndüğünde değişmişti, ben de değişmiştim. Eski dostluğumuz aynı sıcaklıkta sürmedi, ama biz hâlâ dosttuk.',
    'Köy çocukları benim için bambaşka bir dünyaydı. Şehir çocuklarından farklıydılar, daha doğal, daha gerçek. Hile bilmezlerdi, çıkar bilmezlerdi. Bana sevgilerini saf bir şekilde gösterirlerdi. Onları her zaman özlüyorum.',
    'İhtiyar bir adam köyün ucundaki tek katlı evde yaşardı. Kapısı her zaman açıktı. Ben yanından geçerken her keresinde "Gel hocam, bir çay iç" derdi. Ara sıra gerçekten girer, onunla otururdum. Konuşmazdık çok, sadece çay içerdik. Bu da bir dostluktu.',
    'Kasabaya yeni gelen bir doktor vardı. Yirmili yaşlardaydı, henüz mezun olmuş. Birkaç ay sonra gitti, daha büyük bir hastaneye atandı. Ondan önce de bir doktor vardı, o da gitmişti. Yıllarca böyle olmuş, kasabada kimse uzun kalmamış.',
    "Akşam üstü kahveye gittim. Birkaç adam tavla oynuyordu, birkaçı sohbet ediyordu. Köşede oturan yaşlı bir dede beni çağırdı yanına. Çay ısmarladı, sonra başladı anlatmaya. Birinci Cihan Harbi'nde nasıl asker olduğunu, nasıl Kafkasya'da savaştığını. Dakikalar bana saatler gibi geldi.",
    'Köyün muhtarı bir gün bana gelmiş, derdini açtı. Bir aile kavgası vardı, çözüm istiyorlardı. Ben hâkim değildim, fakat onlara yardım etmeye çalıştım. Bir saat sonra mesele çözüldü, herkes el sıkıştı. O an ben de bu kasabanın bir parçası olmuştum.',
    'Köyün kuyusu meydanın ortasındaydı. Sabah akşam kadınlar toplanırdı orada. Su çekerken birbirleriyle konuşurlardı, mahallenin haberleri orada konuşulurdu. Ben pencereden bakardım, dudaklarımdan tebessüm yayılırdı.',
    'Hayatımı sorgulamaya başladığımda yirmi beş yaşımdaydım. "Ne yapıyorum ben?" diye sordum kendime. Cevap bulamadım. Bir kasabaya gittim, orada çalışmaya başladım. Belki cevabı orada bulurdum diye düşündüm.',
    'İnsan en çok kendi kendine yalan söyler. Bir karara varır, sonra kararına inanır gibi yapar. Ben de yıllarca yaptım bunu. Şehrin gürültüsünü sevdiğimi söyledim, hâlbuki kasabaya kaçmak istiyordum hep. Sonunda kabullendim, kasabaya geldim.',
    'Bir öğretmenin hayatı zor değildir, ama tatmin edici de değildir maddi açıdan. Manevi açıdan ise zenginlikler verir insana. Bu kasabada öğrettiğim çocuklar bir gün büyüyecek, hayatın içine karışacak. Onların başarısı benim de başarım olacak.',
    'Yıllar geçtikçe insan değer bilmeye başlıyor. Gençken her şey önemsiz gelirdi, anneme babama yeterince zaman ayırmıyordum. Şimdi pişmanım. Onlara ne kadar borçlu olduğumu o zaman fark edememiştim, geç oldu artık.',
    'Hayatımı geri dönüp baktığımda her seçimden pişman değilim, ama bazılarından üzgünüm. O kasabada kalmak yerine şehre dönmüştüm, belki yanlış olmuştu. Sevdiğim insanı bırakmıştım, o da yanlıştı. Fakat yanlışlar olmadan doğrular nasıl anlaşılır?',
    'Mutluluk dedikleri şey aslında küçük şeylerde gizlidir. Sabah taze ekmek kokusu, çocukların gülümsemesi, akşam dostlarla bir çay. Bunlar mutluluk. Büyük olaylar değil, küçük anlar.',
    'Hayatın anlamı yoktur belki, ama anlamı kendimiz veriyoruz ona. Ben öğretmenliği seçtiğimde anlamım belli olmuştu. Çocuklara öğretmek, onlara faydalı bir şey vermek, bu benim için yeterli sebep oldu yaşamaya.',
    "Bir yıl önce hayatımı değiştirmek istemiştim. Şehirden ayrılıp Anadolu'ya geldim. Birçoğu bana deli dedi, kimse anlamadı. Şimdi anlıyorlar mı bilmem, ama ben anlıyorum kendimi. Hayatımın en doğru kararıydı bu.",
    'İnsan yalnız yaşamayı öğrenmeli. Yanında biri olabilir, olmayabilir. Önemli olan kendi başına ayakta kalabilmektir. Ben de yıllarca yalnız yaşadım, kendime yetmeyi öğrendim. Şimdi kim gelirse gelsin, kim giderse gitsin, ben dururum.',
    'Anadolu insanını yıllarca yaşayarak tanıdım. Onların sade hayatı, içten samimiyeti, sıkıntılarına rağmen ayakta durmaları bana her zaman ders oldu. Şehirde hiçbir zaman bulamadığım bir şey, burada, bu kasabalarda gizliydi.',
    'Hayatta her şey değişir, fakat bazı şeyler asla değişmez. İnsanın iç dünyası, onun değerleri, gönlünde sakladığı sevgiler. Bunlar yıllarca aynı kalır. Yaşlanırız, dış görünüşümüz değişir, fakat içimizdeki o çocuk hep aynı kalır.',
]

print(f'Resat Nuri Guntekin: {len(RESAT_NURI)} metin')

# Yazar 5: Ahmet Rasim (1865-1932)
# Eski Istanbul, gazete kosesi, gozlem, mizah, sokak hayati
AHMET_RASIM = [
    "Eski Beyoğlu'nu hatırlıyorum. O zamanlar şimdiki kadar kalabalık değildi. Cadde boyunca eski binalar dururdu, dükkânlar küçüktü, ama her birinin bir hikâyesi vardı. Akşamları sokak lambaları yanardı, garip bir sıcaklıkla aydınlanırdı orta yer.",
    'Galata kulesinin etrafındaki sokaklar bir labirent gibiydi. İniş çıkışlı yollar, dar geçitler, eski Cenova evleri. Yıllarca o sokaklarda dolaştım, her köşeyi ezbere bilirdim. Şimdi ne zaman geçsem, o eski günleri hatırlayıp dururum.',
    "Cumartesi gecesi Boğaziçi'ne çıkmak başlı başına bir keyifti. Yalıların ışıkları yanardı, vapurların düdükleri uzaktan duyulurdu. Sahillere oturulurdu, bir nargile çekilirdi, akşamın o tatlı serinliğinde dünyayı unuturduk.",
    "Vefa'nın o eski sokaklarında dolaşmak başka bir zevkti. Mahalle bakkalları, esnafın selamlaşması, çocukların sokakta oynaması. Hepsi bir tabloydu, geçen zamanın hatırasıydı. Şimdiki nesil bu mahalleleri bilmiyor, çok yazık.",
    "Kapalıçarşı'nın labirent gibi sokakları yüzlerce yıllık bir geçmişe tanıklık ederdi. Her dükkânda bir esnaf, her esnafın bir hikâyesi vardı. Müşteri gelirdi, pazarlık başlardı, çay servisi yapılırdı. Alışveriş bir tören gibi yapılırdı eskiden.",
    'Şehzadebaşı sokakları aksamlarına kadar canlı olurdu. Tiyatrolardan, kahvehanelerden, meyhanelerden insanlar akıp giderdi. Her birinin elinde bir gazete, bir konuşma. Şehir hayatı orada yoğunlaşırdı, biz de tam ortasındaydık.',
    'Sirkeci istasyonu eskiden çok farklıydı. Trenler buharlı locomotiflerle gelirdi, peronda toz duman olurdu. Yolcular ellerinde bavullarla koşardı. Bir kahvenin önünde durup bu telaşı seyretmek başlı başına bir eğlenceydi.',
    "Eski Tatavla'nın kendine has bir havası vardı. Rum nüfusu yoğundu, kiliseler vardı, manavlar Yunanca konuşurdu. Akşamları meyhaneler dolardı, hafif müzik yayılırdı dışarıya. Şimdi orası başka bir yer oldu, o eski ruh kalmadı.",
    "İstanbul'un eski mahalleleri her biri ayrı bir köy gibiydi. Aksaray başka, Topkapı başka, Eyüp başka. Her mahallenin kendi insanı, kendi kahvesi, kendi camisi vardı. Şimdi hepsi birbirine karıştı, eski sınırlar silindi.",
    'Galata köprüsü üstünde yürümek bütün şehri bir bakışta görmek gibiydi. Tek yandan Boğaz, öbür yandan Haliç. Üzerinden vapurlar geçer, balıkçılar olta atardı. Köprünün altından eski kayıkçılar geçirirdi yolcuları. Hepsi gitti şimdi.',
    'Beşiktaş çarşısı sabahın erken saatlerinde uyanırdı. Manavlar mallarını dizerdi, kasaplar dükkânlarını açardı. Kahveler doluşurdu işine giden esnafla. Akşam üstü çarşı boşalırdı, herkes evine dönerdi. Bu döngü her gün tekrar ederdi yıllarca.',
    "Eski Üsküdar'ı hatırlıyorum. Sahili boyunca yalılar dizilirdi, bahçelerinde meyve ağaçları olurdu. Vapurdan çıkanlar bu yalıların önünden geçerek evlerine ulaşırdı. Şimdi yalıların çoğu yok, yerlerini apartmanlar aldı. İstanbul'un yarısı değişti.",
    'Akşam meyhanesine girdiğimde sıcak bir hava karşıladı beni. Müşteriler masalarda otururdu, garsonlar koşardı bir oraya bir buraya. Tezgâhın arkasında usta, kadehleri dolduruyordu. Köşede bir köşede ben her zamanki yere oturdum, garson "Hoş geldin" dedi.',
    'Bizim meyhane çok eskiydi, kuruluşu Sultan Aziz devrine kadar gidiyordu. Duvarlarda eski fotoğraflar, tavanda yıllarca biriken sigara dumanı izleri. Müşterilerin çoğu yıllardır oradaydı. Yeni gelenler de zamanla aileye katılırdı.',
    'Beyoğlu gecelerinin başka bir tadı vardı. Saat dokuzdan sonra cadde insanlarla dolardı. Hanımlar, beyler kol kola yürürdü. Pastacılar, kahvehaneler, meyhaneler ışıklarını yakardı. Şehir gerçekten yaşar gibi olurdu o saatlerde.',
    'Eski meyhanelerin müşterileri kabadayı değildi, edebiyatçıydı, sanatçıydı. Gece yarısına kadar şiir okunurdu, fikir tartışılırdı. Rakı içilirdi, ama ölçüsünü kaybeden çıkmazdı. Bir adabı vardı her şeyin, yıllarca süren bir geleneği.',
    "Galatasaray'da bir meyhane vardı, bütün gazeteciler oraya giderdi. Sabaha kadar oturulurdu, ertesi günün gazetesi orada planlanırdı. Patronu bizi tanırdı, masaları bizim için ayırırdı. Akşam saatlerinde girdiğimde mutlaka bir tanıdık otururdu.",
    'Sokakta sarhoş bir adam yürüyordu, kendi kendine konuşuyordu. Köşedeki bekçi onu kollarına alıp eve götürmek istedi. Adam karşı koydu "Bırak beni, yürüyebilirim!" diye bağırdı. Sonra düşmek üzereyken bekçi yine yakaladı, eve teslim etti.',
    "Beyoğlu'nun pastanesi ünlüydü. Hanımlar öğleden sonra oraya giderdi. Tatlı çayını içerken son moda kıyafetlerden, son haberlerden konuşurlardı. Pencereden geçenleri seyrederlerdi. Beyoğlu çay saatleri vazgeçilmezdi.",
    "Tarihi bir lokantaya gittim akşam. Beyoğlu'nun ortasında, üst katındaydı. Kapısında bir uşak vardı, ceketini alır asardı. İçerisi loş, mumlu masalar. Yemekler nefisti, müzik hafifti. Bir tatil havasıydı orada akşam geçirmek.",
    "Galata'da bir bar vardı, denize bakar pencereleri olan. Müşteriler oraya akşamleyin giderdi. Bir kadeh likör içerken Boğaz manzarasını seyrederdi. Bahar akşamları, sonbahar akşamları, her mevsim ayrı güzelliği vardı o yerin.",
    "Eski bir kabare vardı Pera'da. Avrupa'dan gelen sanatçılar orada sahne alırdı. Hanımlar şarkı söylerdi, dans ederdi. Müşteriler şampanya içerdi, sigaralarını yakardı. Bütün Avrupa'nın havası vardı o salonda.",
    'Garson eski müşterilerinden birine yaklaştı. "Bey efendi her zamanki gibi mi?" diye sordu. Adam başını salladı. Beş dakika sonra masada her zamanki yemek vardı. Otuz yıllık bir ritüel bu, hiç değişmiyor.',
    'Meyhanenin köşesinde her akşam aynı adam otururdu. Yalnız, sessiz, bir kadeh karşısında. Kimse ona ne yaptığını sormazdı, o da kimseye karışmazdı. Yıllarca böyle gitmişti. Bir gün kayboldu, bir daha gelmedi, kimse de aramadı.',
    'Tiyatro çıkışında insanlar yola dökülürdü. Hanımlar şallarını omuzlarına alır, beyler bastonlarını uzatırdı. Bir kısmı eve giderdi, bir kısmı bir kafeye uğrar gece muhabbetini tamamlardı. Beyoğlu o saatlerde tam canlanırdı.',
    'Sabahın köründe bir simitçi sesi duyuldu. "Sımıt!" diye bağırıyordu, dışarıda yağmur yağmasına rağmen. Pencereyi açıp baktım, üstü başı ıslanmış küçük bir çocuk. Geçim derdi, hava şartlarına aldırmaz.',
    'Mahalle bakkalı yıllarca aynı dükkânda işini yapardı. Bütün mahalleliyi tanırdı, isimleriyle hitap ederdi. Veresiyeyi defteri vardı, herkes defterine yazardı borcunu. Aybaşında borçlar ödenir, defter sıfırlanırdı. Şimdi böyle güven yok artık.',
    'Sokaktaki ayakkabı boyacısı çocuktu, on yaşında olmalıydı. Sabahtan akşama kadar müşteri beklerdi köşede. Bir lira için ayakkabı parlatırdı, sonra çayını içerdi. Ben her geçtiğimde ona bir bahşiş bırakırdım, gözlerinde sevinç görmek beni mutlu ederdi.',
    'Manav Halil Efendi yıllardır aynı dükkândaydı. Sabah erken kalkardı, malını dizerdi. Müşteri geldiğinde "Hoş geldin" derdi sıcakkanlılıkla. Elması kendi köyünden, üzümü kendi bağından getirirdi. Onun manavından alışveriş bir tören gibiydi.',
    'Eski sucu vardı mahallede, sabah akşam su dağıtırdı. Bir at arabasıyla gelir, fıçılarını boşaltırdı. "Sucu geldi!" diye seslenirdi sokakta. Hanımlar pencerelerden bakar, kovalarını uzatırdı. Su pahalıydı, ama hayati bir ihtiyaçtı.',
    'Sokak çalgıcısı her gün aynı köşede otururdu. Sazıyla türküler söylerdi. Yanından geçen biri metelik bırakırdı önündeki kasaya. Akşamleyin para sayardı, ekmek alır eve dönerdi. Hayat zor, ama o şarkı söylemekten vazgeçmezdi.',
    "Köfteci Salih Usta'nın dükkânı çok meşhurdu. Öğle vakti müşterilerle dolardı. Mangalın başında dururdu, köfteleri çevirirdi. Müşteriler ayakta yerdi, çay içerdi. Yarım saatte bir öğün, sonra herkes işine giderdi. Bu o zamanın hızıydı.",
    'Çarşıda bir baharatçı vardı, bütün baharatları yığın yığın sergilerdi. Karabiber, kimyon, kekik, nane. Kokular havayı doldururdu. Kadınlar gelir, koklayarak alırdı. Esnaf onlara nasıl kullanılacağını anlatırdı sabırla. Bu da bir kültürdü.',
    'Sokakta limonata satan bir adam vardı. Sıcak yaz günlerinde elindeki tepsiyle dolaşırdı. "Buz gibi limonata!" diye bağırırdı. Çocuklar koşarak yanına gelirdi. Birer kuruşla bir bardak içerlerdi, sonra dağılırlardı.',
    'Eski terzi dükkânına girdim bir gün. Yaşlı terzi gözlüklerinin üstünden bana baktı. Ona elbisemi nasıl istediğimi anlattım. O dikkatle dinledi, ölçü aldı, bir hafta sonra gel dedi. Bir hafta sonra elbise hazırdı, kusursuzdu. Şimdi böyle terziler kalmadı.',
    'Mahalle berberi sabah dükkânı açardı, ilk müşteri kapıdan girerdi. Tıraş yaparken aynı zamanda dedikodu yapardı. Mahalleli haberleri orada öğrenirdi. Müşteri gittikten sonra başka biri gelirdi, döngü böyle sürerdi sabahtan akşama kadar.',
    'Eski seyyar satıcılar yok artık. Eskiden sokakta tencere tava satanı, çay satanı, dondurma satanı, her şey satanı görürdün. Hepsi kendi tarzında bağırırdı malını. Her birinin sesini çocukluğumdan hatırlıyorum, kulaklarımdan silinmedi.',
    'Lokum satıcısı tepsisini başında taşırdı sokakta. "Tatlı lokum, gül lokumu, fıstıklı lokum!" diye bağırırdı. Çocuklar onun arkasında koşturuyordu. Bir kuruş veriliyordu, bir lokum alınıyordu. O lokumun tadı dünyalara değer geliyordu o yaşta.',
    'Gazete bürosunda akşamleyin işler ısınırdı. Sabahki sayı için son düzeltmeler yapılırdı. Editör en son haberleri seçerdi. Yazarlar köşelerini son anda yetiştirirdi. Matbaada baskı makineleri sabaha kadar çalışırdı. Bir gazete bir tiyatro gibiydi.',
    'Gazete yazarlığı kolay değildir. Her gün bir köşe yazısı, bir konu, bir fikir. Boş kafayla otururken bile yazmak zorundasın. Ben yirmi yıldır bunu yapıyorum. Bazen feyiz gelir kalemden döküleyazılar, bazen iki cümleyi zar zor yan yana getirirsin.',
    'Köşe yazıma her sabah başlardım, bittiğinde gazeteye giderdim. Editörle konuşurduk biraz, sonra eve dönerdim. Akşamleyin bir kahve, sonra meyhane, sonra ev. Bu döngü yıllarca sürdü, hiç sıkılmadım, çünkü her gün yeni bir şey yaşıyordum.',
    'Bizim zamanımızda gazete farklıydı. Şimdi gibi reklam dolu değildi, haber doluydu. Yazarlar gerçekten yazardı, hayat hakkında düşünür, fikir üretirdi. Şimdi gazeteler sadece bir tüketim ürünü oldu, ne yazık.',
    'Bir köşe yazarı zamanın aynası olmalıdır. Sokakta olanı, kahvede konuşulanı, evde tartışılanı yazıya dökecek. İnsanlar gazete açtığında "İşte bu adam beni anlıyor" desin. Yıllarca bu hedefe ulaşmaya çalıştım.',
    'Yazı yazmak ağır bir iştir. İnsan sürekli kendisinden bir şeyler verir. Yıllarca bunu yapan kişi yıpranır, yorulur. Bazen kalem elimden düşer, devam edemem. Sonra biraz dinlenir, tekrar başlarım. Yazı yazmak bir tutkudur, bir hastalık gibi.',
    'Gazete bayilerin başında durup okuyanları seyrediyordum. Bir adam gazete almıştı, ayakta hızlı hızlı okuyordu. Sonra başka biri geldi, omuzunun üstünden okudu. Üçüncüsü geldi, o da okudu. Bir gazete üç kişiyi mutlu ediyordu. Bu da bir başka mutluluk.',
    'Eski yazarlar arasında bir samimiyet vardı. Akşamları bir kafede buluşurduk, sabaha kadar otururduk. Fikirler tartışılır, yeni eserler okunur, eleştiriler yapılırdı. O dostluklardan beslenirdik hepimiz. Şimdiki yazarlar bunu bilmiyor, herkes kendi köşesinde.',
    'Matbaada akşamleyin koşturmaca başlardı. Mürettipler dizgiyi yapardı, baskıcılar makineyi hazırlardı. Gece yarısına doğru ilk nüshalar çıkardı. Editör bir kontrol yapardı, sonra basıma izin verirdi. Sabah erkenden bayiler önünde gazete olurdu.',
    'Genç bir gazete yazarına nasihat verirken hep aynı şeyi söylüyorum. "Sokağa çık" diyorum, "insanlarla konuş, hayatı yaşa, sonra yaz. Masa başında yazılan yazı kuru olur. Yaşamadan yazamazsın." Bu sözü dinleyenler iyi yazar olmuştur hep.',
    'Eski bayramların tadı başkaydı. Bir hafta öncesinden hazırlıklar başlardı. Evler temizlenirdi, yeni elbiseler dikilirdi, bayram tatlıları pişirilirdi. Bayram sabahı namazdan sonra herkes birbirini ziyaret ederdi. Üç gün boyunca bu sürerdi.',
    'Bizim evimizde ramazan ayrı bir önem taşırdı. İftar sofrası özenle hazırlanırdı, davetliler gelirdi. Sahurda bütün aile bir araya gelirdi. Top patladığında oruç açılırdı, herkes sofraya oturardı. Bu birliktelikler yıllarca hatırlanırdı.',
    'Eski düğünler bir hafta sürerdi. Önce kına gecesi yapılırdı kızlar için. Sonra erkek tarafının töreni olurdu. Düğün günü gelin alma, sonra düğün yemeği. Mahalleli, akrabalar, dostlar hep katılırdı. Şimdi düğün bir akşamda biter, eski tat yok.',
    'Cumartesi günleri hamama gitmek bir ritüeldi. Kadınlar erkenden giderdi, akşama kadar kalırdı. Birbirleriyle konuşur, türkü söyler, eğlenirlerdi. Tellak hizmet ederdi, kese yapardı. Çıktıklarında bir saat kahve içerlerdi, sonra eve dönerlerdi.',
    'Eski sünnet törenleri büyük olaylardı. Çocuğa atlas elbise giydirilirdi, başına süslü kep takılırdı. At sırtında mahalleyi gezerdi. Sonra düğün yemeği yapılırdı. Akrabalar hediye getirirdi, çocuk yüzlerce hediyeyle uyanırdı.',
    'Mahallenin akşam ezanı bir saatti adeta. Bütün esnaf dükkânını kapatırdı, evine giderdi. Sokaklar bir anda boşalırdı. Camiler dolardı, namaz kılınırdı. Sonra herkes sofrasına otururdu. Bu saat sessizliği şehrin ritminin parçasıydı.',
    'Eski mevlitler çok ihtişamlı olurdu. Bütün mahalle camiye toplanırdı. Mevlithanlar tatlı sesleriyle okurdu. Lokmalar dağıtılırdı, şerbet ikram edilirdi. Sonra evlerde de devam ederdi tören. Bu maneviyat dolu günler unutulmazdı.',
    'Eski cuma günleri özel bir öneme sahipti. Sabah hamama gidilirdi, sonra cuma namazına. Akşam aileler ziyaret edilirdi. Bu dini bir tören gibiydi, ama aynı zamanda sosyal bir gelenek. Bütün hafta bu günü beklenirdi.',
    'Çocukken bayramda büyüklerin elini öperdik. Onlar bize harçlık verirdi, bayram şekeri ikram ederdi. Bir günde bütün mahalleyi dolaşır, ceplerimiz dolardı parayla. Akşam babamız bu paranın yarısını kumbaramıza koydururdu, ekonomi dersini öyle alırdık.',
    'Eski mahalle imamı sabah ezanını okurdu, sonra mahalleyi dolaşırdı. Bütün evlerin önünden geçer, hâl hatır sorardı. Hastalar varsa onları ziyaret ederdi. Bu tek başına bir sosyal vazifeydi, mahallenin manevi rehberiydi imam.',
    'Bir tramvayda bir adamın diğerine küfretmesi normal sayılırdı eskiden değil. Saygı önemliydi, çevredeki kadınlar düşünülürdü. Şimdi insanlar açıkça küfrediyor, sigarayı tramvayda içiyor, ayağını koltuğa atıyor. Şehir hayatı bayağılaştı.',
    'Şehirde her şey hızlandı, fakat bu hız insanları mutlu etmedi. Eskiden insanlar yavaş yaşardı, ama yaşadıkları her ânın tadını alırdı. Şimdi hız var, ama keyif yok. İnsanlar nereye koşuyor anlamak zor.',
    'Eski İstanbullular kıyafetlerine çok dikkat ederdi. Bir adam ceketsiz, kravatsız sokağa çıkmazdı. Hanımlar şapkasız dışarı adım atmazdı. Bu sadece bir kibarlık değil, kendine saygıydı. Şimdi insanlar sokakta pijama ile dolaşıyor, kıyafet kültürü yok oldu.',
    'Vapurda her zamanki gibi insanları seyrediyordum. Karşımda bir hanım kitap okuyordu. Yanımda iki bey siyaset tartışıyordu. Arkamda bir çocuk ağlıyordu, annesi onu susturmaya çalışıyordu. Vapur küçük bir toplum gibiydi, her tipten insan vardı.',
    'Modernleşme dediğimiz şey her şeyimizi değiştirdi. Eski adetleri unuttuk, yeni adetler de tam yerleşmedi. İki kültür arasında kaldık. Yeniyi tam alamadık, eskiyi de bırakamadık. Bu kararsızlık zamanla geçecek belki, ama biz onu göremeyeceğiz.',
    'Eski mahalle hayatı bir aileydi. Komşular birbirini tanırdı, yardımlaşırdı. Bir derdi olan komşuya gidilirdi, bir mutluluğu olan kutlanırdı. Şimdi yan dairede oturanı bile tanımıyoruz. Apartman hayatı insanı yalnızlaştırdı.',
    'Genç nesil kitap okumaktan vazgeçti. Eskiden kahvelerde gazete okunurdu, evlerde kitap. Şimdi herkes telefona bakıyor. Bu değişimin sonuçlarını yıllar sonra göreceğiz. Belki çok geç olmadan bir şey yapılır, umarım.',
    'Şehir büyüdükçe küçüldü ruhumuz. Eskiden komşularımızı tanırdık, sokakta selamlaşırdık. Şimdi milyonlarca insan içinde yalnızız. Şehir büyüdü, ama biz küçüldük. Bu modernliğin paradoksu.',
    'Cuma günleri esnafın çoğu camiye giderdi, dükkân kapanırdı bir saatliğine. Müşteriler beklemek zorundaydı, kimse şikâyet etmezdi. Şimdi her şey kâr odaklı, hiçbir şey için duraklamıyoruz. Maneviyat zaman aldığı için verimsiz sayılıyor.',
    'Şehrin değişimini yıllardır izliyorum. Beton yığınları her yıl artıyor, yeşil alanlar azalıyor. Çocuklar oyun oynayacak yer bulamıyor. Sokakta top oynayan çocuk azaldı, çünkü sokak yok artık, sadece otomobil.',
    "Boğaziçi'nde mehtap doğduğunda her şey değişirdi. Sular gümüş bir tabaka kaplanırdı. Yalıların pencereleri parıldardı. Vapurlar yavaşça geçerdi, ışıklarını yansıtarak. O manzara gördükçe bir şair olası geliyordu insanın.",
    "Adalardan dönen vapur akşamleyin Sirkeci'ye yanaşırdı. İnsanlar bavullarıyla, sepetleriyle iniyordu. Bir yaz gününü tamamlamış yüzlerinde rahatlık vardı. Garson yorgun, ama memnun bir şekilde toplardı son fincanları.",
    "Sahile inip sigara içerken vapurların geçişini seyretmek başka bir keyifti. Düdükleri uzaktan gelirdi, ışıkları suya yansırdı. Bir vapur Eminönü'ne, biri Üsküdar'a, biri Adalar'a giderdi. Hepsi bir senfoni yapardı sularda.",
    'Galata köprüsünde gece yarısı yürürdüm bazen. Şehir sessizleşmiş, sadece dalgaların sesi duyuluyordu. Köprünün altında balıkçılar otururdu, oltalarını atarlardı. Sabaha kadar orada kalırlardı, soğuğa rağmen.',
    "Akşam ezanı Süleymaniye'den yükselince bütün şehir bir an dururdu. Ezan sesi havayı dolaşırdı, bütün camilerden cevap gelirdi. Sokakta yürüyenler bile duraklardı bir an. Bu sesleri duymak şehrin manevi nabzını hissetmekti.",
    'Eski Boğaz vapurları masallar gibiydi. Geniş güverteleri olurdu, salonlarında piyano bulunurdu. İçinde garsonlar gezerdi, çay servisi yapardı. Üst güvertede oturmak başlı başına bir keyifti. Manzara, hava, rahatlık. Şehirden çıkmadan tatil gibiydi.',
    'İstanbul gecelerinde yıldızlar görünmez şehrin ışıklarından, ama denizin yüzünde yansıyan ışıklar yıldız yerine geçer. Boğaz boyunca yürürken hep ışıklar göreceksin. Yalılarda, vapurlarda, sokak lambalarında. Şehir hiç uyumaz.',
    'Bir yaz akşamı Anadolu yakasında bir bahçede oturuyorduk. Karşı kıyıda İstanbul ışıklarıyla parlardı. Ufukta minareler dikilirdi, vapurlar geçerdi. Hayat ne kadar güzeldir o anlarda. Sonra eve döneriz, gündelik dertler başlar tekrar.',
    "Galata kulesinden bakan İstanbul'u görür. Bir tarafta Haliç, diğer tarafta Boğaz. Üç deniz bir araya gelirdi bakışında. Saraylar, camiler, kiliseler, evler. Tarih ve bugün iç içe. Bu manzarayı görenler İstanbul'u sevmemezlik edemez.",
    'Akşam vapuruyla evime dönerken denize bakardım. Sular gri, sis hafif. Karşı kıyı görünmez bazen. O an dünyadan kopmuş gibi olurdum. Yalnız ben, vapur ve deniz. Bütün şehri arkamda bırakmış, yeni bir günlük hayata gidiyormuş gibi hissederdim.',
    'Komşumuzun karısı her gün pazara giderdi, akşam eli boş dönerdi. Eşi sorardı: "Hanım, ne aldın?" "Hiçbir şey beğenmedim" derdi. Adam sabırla beklerdi. Bir hafta bu sürerdi, sonunda hanım bir şeyler alırdı. Pazarcılık başlı başına bir sanat, hanımlık ondan büyük bir sanat.',
    'Mahallenin avukatı her sorunda "Ben hallederim" derdi. Halletmezdi, ama söz verirdi. Müvekkilleri yıllarca beklerdi, davaları sürüncemede kalırdı. Ne tuhaf, herkes onun gerçek niyetini biliyordu, ama yine de ona giderlerdi. Belki başka kimseleri yoktu.',
    'Bir adam zenginleşince mahalleden ayrılır, semt değiştirir. Birkaç yıl sonra geri gelir eski mahalleye, herkese kibirle bakar. Mahalleli güler arkasından. "Bu adamı küçükken biliyoruz" derler. "Hâlâ çocuk, sadece para sahibi olmuş."',
    'Ben gazetede çalışırken bir okuyucu sürekli mektup yazardı. Her yazımı eleştirirdi, her köşemde hata bulurdu. Bir gün buluşma talep etti. Geldi, masanın karşısına oturdu. "Yıllardır sizi okuyorum" dedi. "Beğeniyor musunuz?" diye sordum. "Hayır" dedi. Yine de okuyordu.',
    'Mahallenin softa adamı sabahtan akşama kadar herkesin ahlakını eleştirir. Ama akşam bir bakarsın, kendisi kahveye gitmiş tavla oynuyor, çay içiyor. Kimseye sezdirmez bunu. Halka vaaz veren bir başka, kendisi başka. İnsan hâli işte, ikiyüzlülük damarımıza işlemiş.',
    'Bir gün bir adam yanıma geldi, hayat hikâyesini anlatmak istedi. Bir saat dinledim. Sonra hesap istedim ondan. "Ne hesabı?" dedi şaşkın. "Beni bir saat dinlettin, vaktimi yedin. Bir saatim para eder" dedim gülerek. O da güldü, sonra kalktı gitti.',
    'Eski bir ahbabıma rastladım yıllar sonra. Çok kibirlenmişti, kendinden bahsediyordu durmadan. Beni dinlemiyordu bile. Bir saat sonra "Söyle, ne yapıyorsun?" dedi. "Sizi dinliyorum" dedim. Anlamadı ironiyi, devam etti kendinden bahsetmeye.',
    'Komşu kadın gelinine sürekli takılırdı. Halbuki kendisi de zamanında nasıl gelindi unutmuştu. Gelinine "Benim zamanımda gelinler dik dururdu" derdi. Aynaya baksaydı gençliğindeki kayınvalidesinin nasıl bir gelin olduğunu hatırlardı. İnsan unutuyor, bencillik unutturuyor.',
    'Bizim eski bir tanıdık siyasete girmişti. Önce halkın hizmetinden bahsederdi, sonra kendi cebine bakardı. Mahalleli farkına vardı bunun, ama söylemedi. "Hepsi böyledir" dediler. Belki haklılardı, belki değil. Önemli olan, kimsenin bir şey değiştirmek için adım atmamasıydı.',
    'Bir doktor vardı mahallede, herkesin hastalığını okşamadan tedavi ederdi. Hasta ne derse desin "Sen merak etme, ben anladım" derdi. Çoğu zaman yanılırdı tabii, ama özgüvenle çalışıyordu. İlginç bir şekilde hastalarının çoğu iyileşirdi, belki güveni hastalığı atıyordu.',
]

print(f'Ahmet Rasim: {len(AHMET_RASIM)} metin')

# Butun yazarlari tek bir DataFrame'de topla
yazarlar = {
    0: ('Omer Seyfettin', OMER_SEYFETTIN),
    1: ('Sabahattin Ali', SABAHATTIN_ALI),
    2: ('Huseyin Rahmi', HUSEYIN_RAHMI),
    3: ('Halit Ziya', HALIT_ZIYA),
    4: ('Resat Nuri', RESAT_NURI),
    5: ('Ahmet Rasim', AHMET_RASIM),
}

veriler = []
for yid, (isim, metinler) in yazarlar.items():
    for metin in metinler:
        veriler.append({'metin': metin, 'yazar': isim, 'etiket': yid})

df = pd.DataFrame(veriler)
print(f'Toplam orijinal metin sayisi: {len(df)}')
print('\nHer yazardan kac metin var:')
print(df['yazar'].value_counts())

df.head(3)

# Yazar basina metin sayisi - gorselleme
fig, ax = plt.subplots(figsize=(10, 5))
counts = df['yazar'].value_counts()
ax.bar(counts.index, counts.values,
       color=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c'])
ax.set_xlabel('Yazar')
ax.set_ylabel('Metin sayisi')
ax.set_title('Orijinal veri seti - Yazar basina metin sayisi')
plt.xticks(rotation=20, ha='right')
for i, v in enumerate(counts.values):
    ax.text(i, v + 1, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# Cumle siralama augmentation
import re

def cumlelere_ayir(metin):
    """Metni cumlelere ayir."""
    cumleler = re.split(r'(?<=[.!?])\s+', metin.strip())
    return [c.strip() for c in cumleler if c.strip()]

def cumle_siralarini_degistir(metin, rng):
    """Bir metnin cumle sirasini rastgele degistir."""
    cumleler = cumlelere_ayir(metin)
    if len(cumleler) < 2:
        return None  # tek cumlede augment yapamayiz
    karistirilmis = cumleler[:]
    # Orijinalden farkli olmasi icin garanti
    while karistirilmis == cumleler:
        rng.shuffle(karistirilmis)
    return ' '.join(karistirilmis)

# Augmentation uygula
rng = random.Random(SEED)
augmented_rows = []
for _, row in df.iterrows():
    yeni = cumle_siralarini_degistir(row['metin'], rng)
    if yeni is not None and yeni != row['metin']:
        augmented_rows.append({
            'metin': yeni,
            'yazar': row['yazar'],
            'etiket': row['etiket']
        })

df_aug = pd.DataFrame(augmented_rows)
print(f'Augmentation ile uretilen yeni metin: {len(df_aug)}')

# Orijinal + augmented birlestir
df_full = pd.concat([df, df_aug], ignore_index=True)
df_full = df_full.sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f'\nGenisletilmis veri seti boyutu: {len(df_full)}')
print('\nYazar dagilimi:')
print(df_full['yazar'].value_counts())

# Augmentation ornegi
ornek_idx = 0
orig = df.iloc[ornek_idx]['metin']
augmented_example = cumle_siralarini_degistir(orig, random.Random(SEED))

print('ORIJINAL:')
print(orig)
print('\nAUGMENTED:')
print(augmented_example)

# Etiketler zaten sayisal (df_full['etiket'])
le = LabelEncoder()
le.fit(df_full['yazar'])

X = df_full['metin'].tolist()
y = df_full['etiket'].values

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=SEED,
    stratify=y
)

print(f'Egitim seti: {len(X_train)} metin')
print(f'Test seti:  {len(X_test)} metin')

# id2label ve label2id sozlukleri (BERT icin)
id2label = {yid: isim for yid, (isim, _) in yazarlar.items()}
label2id = {isim: yid for yid, isim in id2label.items()}
print('\nEtiketler:')
for k, v in id2label.items():
    print(f'  {k} -> {v}')

# TF-IDF
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=3000,
    min_df=2,
    sublinear_tf=True
)
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

print(f'TF-IDF matris boyutu: {X_train_tfidf.shape}')

# Logistic Regression
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
lr.fit(X_train_tfidf, y_train)

y_pred_baseline = lr.predict(X_test_tfidf)
acc_baseline = accuracy_score(y_test, y_pred_baseline)
print(f'\nBaseline test dogrulugu: {acc_baseline:.4f}')
print('\nSiniflandirma raporu:')
print(classification_report(
    y_test, y_pred_baseline,
    target_names=[id2label[i] for i in sorted(id2label.keys())]
))

# Hugging Face'ten Turkce BERT yukle
MODEL_ADI = 'dbmdz/bert-base-turkish-cased'
NUM_LABELS = len(yazarlar)  # 6

tokenizer = AutoTokenizer.from_pretrained(MODEL_ADI)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ADI,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)
print(f'Model yuklendi: {MODEL_ADI}')
print(f'Sinif sayisi: {NUM_LABELS}')

# PyTorch Dataset sinifi
class YazarDataset(Dataset):
    def __init__(self, metinler, etiketler, tokenizer, max_uzunluk=128):
        self.metinler = metinler
        self.etiketler = etiketler
        self.tokenizer = tokenizer
        self.max_uzunluk = max_uzunluk
    
    def __len__(self):
        return len(self.metinler)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.metinler[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_uzunluk,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(int(self.etiketler[idx]), dtype=torch.long)
        }

train_dataset = YazarDataset(X_train, y_train, tokenizer)
test_dataset = YazarDataset(X_test, y_test, tokenizer)

print(f'Train dataset boyutu: {len(train_dataset)}')
print(f'Test dataset boyutu: {len(test_dataset)}')

ornek = train_dataset[0]
print(f'\nOrnek input_ids shape: {ornek["input_ids"].shape}')
print(f'Ornek label: {ornek["labels"].item()} -> {id2label[ornek["labels"].item()]}')

# Egitim parametreleri
training_args = TrainingArguments(
    output_dir='./bert-yazar-tespiti',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy='epoch',
    save_strategy='no',
    report_to='none',
    seed=SEED,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print('Trainer hazir, egitim baslayabilir.')

# Egitim
print('Egitim basliyor...')
trainer.train()
print('\nEgitim tamamlandi.')

# Son degerlendirme
sonuc = trainer.evaluate()
acc_bert = sonuc['eval_accuracy']
print(f'BERT test dogrulugu: {acc_bert:.4f}')

# BERT'in test tahminleri
predictions_out = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions_out.predictions, axis=-1)

print('BERT Siniflandirma Raporu:')
print(classification_report(
    y_test, y_pred_bert,
    target_names=[id2label[i] for i in sorted(id2label.keys())]
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_bert)
yazar_isimleri = [id2label[i] for i in sorted(id2label.keys())]

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=yazar_isimleri,
            yticklabels=yazar_isimleri,
            cbar=True, ax=ax)
ax.set_xlabel('Tahmin / Predicted')
ax.set_ylabel('Gercek / True')
ax.set_title('BERT - Confusion Matrix (Test seti)')
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Iki modeli karsilastir
karsilastirma = pd.DataFrame({
    'Model': ['TF-IDF + LogReg (baseline)', 'BERT (fine-tuned)'],
    'Test Dogrulugu': [acc_baseline, acc_bert]
})
print(karsilastirma.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(karsilastirma['Model'], karsilastirma['Test Dogrulugu'],
       color=['#95a5a6', '#27ae60'])
ax.set_ylim(0, 1.05)
ax.set_ylabel('Test Dogrulugu')
ax.set_title('Model Karsilastirmasi')
for i, v in enumerate(karsilastirma['Test Dogrulugu']):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

iyilesme = (acc_bert - acc_baseline) / max(acc_baseline, 1e-9) * 100
print(f'\nBERT, baseline icin %{iyilesme:.1f} daha basarili.')

def tahmin_et(metin):
    """Verilen metnin hangi yazara ait olabilecegini tahmin et.
    
    Not: model, tokenizer ve id2label degiskenlerini global scope'tan kullanir.
    Bu hucreden once BERT modelinin yuklendigi ve egitildigi hucreler calismis olmali.
    """
    # Modelin hangi cihazda oldugunu ogren (GPU/CPU)
    # Find which device the model lives on (GPU/CPU)
    device = next(model.parameters()).device
    
    model.eval()
    inputs = tokenizer(metin, return_tensors='pt',
                       truncation=True, padding=True, max_length=128)
    # Input tensorlerini ayni cihaza tasi
    # Move input tensors to the same device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # cpu()'ya geri getir ki numpy/python ile calisabilelim
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    return id2label[pred_id], confidence, probs.numpy()


# Test metinleri - her yazarin uslubunda yeni cumleler
test_metinleri = [
    ('Cephe sessizdi, askerler siperde bekliyordu. Birden top sesi geldi, '
     'herkes yere kapaklandi. Komutan bagirdi: Hazirlanin, dusman geliyor!',
     'Omer Seyfettin'),
    
    ('Berlin in soguk gunlerinde pansiyon odasinda yalniz oturuyordum. '
     'Sigaramin dumani lambanin isiginda kayboluyor, ben de kendi '
     'dusuncelerimde kayboluyordum. Memleket hasreti yine icimde uyandi.',
     'Sabahattin Ali'),
    
    ('Naciye Hanim soluk soluga komsuya kostu. Aman komsucugum, duydun mu? '
     'diye haykirdi. Karsi konaktaki Pakize Hanim kahveyi pisirmeye basladi, '
     'dedikodu saat sona kadar bitmeyecekti.',
     'Huseyin Rahmi'),
    
    ('Salonun lof isiginda kristal avizelerin pirimltilari camlardan icer '
     'suzulen aksamin son isiklariyla birlesiyordu. Hanimefendi koltuga '
     'gomulmus, gozleri uzaklarda, gecmis gunlerin solgun bir tablosunu '
     'seyrediyordu.',
     'Halit Ziya'),
    
    ('Kasabaya yeni atandigim gun, otobus duraginda ufak bir bavulla '
     'durdugumu hatirliyorum. Hicbir tanidigim yoktu burada. Mektep nerede '
     'diye sordum bir koyluye, o bana yolu tarif etti, yuruyerek gittim.',
     'Resat Nuri'),
    
    ('Eski Beyoglu nu hatirliyorum. Aksamlari sokak lambalari yanardi, '
     'meyhanelerin kapilari acilirdi. Kaldirimda yuruyen hanimlar, beyler, '
     'kol kola gezenler. Sehir hayati orada yogunlasirdi.',
     'Ahmet Rasim'),
]

dogru = 0
print('=' * 70)
for metin, beklenen in test_metinleri:
    tahmin, guven, _ = tahmin_et(metin)
    isaret = 'DOGRU' if tahmin == beklenen else 'YANLIS'
    if tahmin == beklenen:
        dogru += 1
    print(f'\nMetin: {metin[:100]}...')
    print(f'  Beklenen: {beklenen}')
    print(f'  Tahmin:   {tahmin} (guven: {guven:.2%})  [{isaret}]')
    print('-' * 70)

print(f'\n{dogru}/{len(test_metinleri)} dogru tahmin.')
